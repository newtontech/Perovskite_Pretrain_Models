"""Inventory historical public data and predictions; never train or change inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(set(columns)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return columns, rows


def cleaned_test(train, test):
    """Preserve every retained row and its order, including duplicate observations."""
    train_smiles = {row["SMILES"] for row in train}
    return [row for row in test if row["SMILES"] not in train_smiles]


def validate_folds(splits, n, expected_folds=10):
    trains, valids = splits.get("train_indices"), splits.get("valid_indices")
    if not isinstance(trains, list) or not isinstance(valids, list):
        raise ValueError("Missing fold arrays")
    if len(trains) != expected_folds or len(valids) != expected_folds:
        raise ValueError(f"Expected {expected_folds} train/valid folds")
    counts = [0] * n
    for train, valid in zip(trains, valids):
        for indices in (train, valid):
            if not isinstance(indices, list) or not indices:
                raise ValueError("Fold indices must be nonempty lists")
            if any(type(index) is not int or not 0 <= index < n for index in indices):
                raise ValueError("Invalid or out-of-bounds fold index")
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate fold index")
        if set(train) & set(valid) or set(train) | set(valid) != set(range(n)):
            raise ValueError("Fold overlap or incomplete coverage")
        for index in valid:
            counts[index] += 1
    if any(count != 1 for count in counts):
        raise ValueError("Each row must appear in validation exactly once")
    return list(zip(trains, valids))


def metrics(rows):
    if not rows:
        raise ValueError("Cannot compute empty metrics")
    truth = [float(row["true"]) for row in rows]
    pred = [float(row["pred"]) for row in rows]
    if not all(math.isfinite(value) for value in truth + pred):
        raise ValueError("Nonfinite prediction or target")
    n = len(truth)
    errors = [actual - estimate for actual, estimate in zip(truth, pred)]
    squared = math.fsum(error * error for error in errors)
    mean = math.fsum(truth) / n
    total = math.fsum((actual - mean) ** 2 for actual in truth)
    result = {"n": n, "rmse": math.sqrt(squared / n),
              "mae": math.fsum(abs(error) for error in errors) / n,
              "r2": "" if n < 2 or total == 0 else 1 - squared / total}
    if any(not math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("Nonfinite computed metric")
    return result


def record_id(source, row):
    return f"{source}#row={row}"


def build_outputs(root=ROOT):
    manifests, samples, source_columns = [], [], []
    split_rows, members, fold_rows, metric_rows = [], [], [], []
    warnings = []

    def source_csv(path, seed=None, role=None):
        columns, rows = read_csv(path)
        source = path.relative_to(root).as_posix()
        manifests.append({"source_file": source, "format": "csv", "rows": len(rows),
                          "sha256_bytes": digest(path.read_bytes()),
                          "sha256_normalized": digest(csv_bytes(columns, rows))})
        if role is not None:
            for column in columns:
                if column not in source_columns:
                    source_columns.append(column)
            for number, row in enumerate(rows, 1):
                samples.append({"record_id": record_id(source, number), "source_file": source,
                                "source_row": number, "seed": seed, "role": role,
                                "raw_smiles_id": digest(row["SMILES"].encode("utf-8")),
                                **{f"source:{key}": value for key, value in row.items()}})
        return source, columns, rows

    for seed in range(5):
        base = root / "baselines/datasets" / f"split_seed_{seed}"
        datasets = {}
        headers = []
        for name, role in (("train_pool", "train_pool"), ("test_raw", "test_raw"), ("test", "test_cleaned")):
            _, columns, datasets[role] = source_csv(base / f"{name}.csv", seed, role)
            headers.append(columns)
        train, test, clean = (datasets[role] for role in ("train_pool", "test_raw", "test_cleaned"))
        if headers[0] != headers[1] or headers[1] != headers[2] or cleaned_test(train, test) != clean:
            raise ValueError(f"Seed {seed}: cleaned test differs from raw-SMILES filtering")
        train_set = {row["SMILES"] for row in train}
        for role, rows in datasets.items():
            unique = {row["SMILES"] for row in rows}
            split_rows.append({"seed": seed, "role": role, "n_rows": len(rows),
                               "n_unique_raw_smiles": len(unique),
                               "n_rows_overlapping_train_pool": sum(row["SMILES"] in train_set for row in rows),
                               "n_raw_smiles_overlapping_train_pool": len(unique & train_set),
                               "cleaned_equals_raw_filter": "true"})

        molclr = root / "training/molclr/data/perovskite-resplit" / f"split_seed_{seed}"
        source, _, rows = source_csv(molclr / "train_pool.csv", seed, "molclr_train_pool")
        split_path = molclr / "splits.json"
        splits = json.loads(split_path.read_text(encoding="utf-8-sig"))
        manifests.append({"source_file": split_path.relative_to(root).as_posix(), "format": "json",
                          "rows": "", "sha256_bytes": digest(split_path.read_bytes()),
                          "sha256_normalized": digest(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode())})
        for fold, (train_idx, valid_idx) in enumerate(validate_folds(splits, len(rows))):
            overlap = {rows[i]["SMILES"] for i in train_idx} & {rows[i]["SMILES"] for i in valid_idx}
            fold_rows.append({"seed": seed, "fold": fold, "n_train": len(train_idx),
                              "n_valid": len(valid_idx), "n_overlapping_raw_smiles": len(overlap)})
            if overlap:
                warnings.append(f"MolCLR seed {seed} fold {fold}: {len(overlap)} raw SMILES overlap train/valid")
            for role, indices in (("train", train_idx), ("valid", valid_idx)):
                for index in indices:
                    members.append({"seed": seed, "fold": fold, "role": role,
                                    "source_file": source, "source_row": index + 1,
                                    "source_index_zero_based": index,
                                    "record_id": record_id(source, index + 1)})

    for directory in ("baselines/predictions", "baselines/krfp/predictions_krfp"):
        paths = sorted((root / directory).glob("*.csv"))
        if not paths:
            raise ValueError(f"No prediction files in {directory}")
        for path in paths:
            source, columns, rows = source_csv(path)
            if not {"true", "pred", "split"} <= set(columns) or not rows:
                raise ValueError(f"Missing prediction columns or rows: {source}")
            if any(not row["split"].strip() for row in rows):
                raise ValueError(f"Empty prediction split label: {source}")
            for split in sorted({row["split"] for row in rows}):
                metric_rows.append({"source_file": source, "split": split, "target": "TARGET (residual)",
                                    **metrics([row for row in rows if row["split"] == split]),
                                    "status": "historical_prediction_recomputation"})

    tables = {
        "samples.csv": (["record_id", "source_file", "source_row", "seed", "role", "raw_smiles_id"] +
                        [f"source:{column}" for column in source_columns], samples),
        "file_manifest.csv": (["source_file", "format", "rows", "sha256_bytes", "sha256_normalized"], manifests),
        "split_summary.csv": (["seed", "role", "n_rows", "n_unique_raw_smiles", "n_rows_overlapping_train_pool",
                               "n_raw_smiles_overlapping_train_pool", "cleaned_equals_raw_filter"], split_rows),
        "fold_membership.csv": (["seed", "fold", "role", "source_file", "source_row", "source_index_zero_based", "record_id"], members),
        "fold_summary.csv": (["seed", "fold", "n_train", "n_valid", "n_overlapping_raw_smiles"], fold_rows),
        "historical_metrics.csv": (["source_file", "split", "target", "n", "rmse", "mae", "r2", "status"], metric_rows),
    }
    output = {name: csv_bytes(columns, rows) for name, (columns, rows) in tables.items()}
    summary = {"schema_version": 1, "scope": "existing public baseline CSVs, MolCLR folds, historical predictions",
               "source_files": len(manifests), "sample_rows": len(samples), "molclr_folds": len(fold_rows),
               "metric_groups": len(metric_rows), "warnings": warnings,
               "limitations": ["No retraining or confirmation of manuscript results.",
                               "Raw SMILES hashes are not canonical molecule identifiers.",
                               "Record IDs identify source-file rows, not unique physical observations across seeds or sources.",
                               "Fold indices are zero-based; source rows are one-based data rows, excluding the header.",
                               "Baseline train_pool is not an explicitly frozen inner train/validation assignment.",
                               "Historical fold SMILES overlap is reported, not corrected.",
                               "TARGET residual metrics only; no Delta_pred addback without verified row mapping.",
                               "No best-seed selection. Missing initialization/tuning/conformer seeds are not inferred.",
                               "Source columns are strings; absent columns in the union export are blank.",
                               "Normalized hashes ignore CSV line endings/BOM and JSON formatting; byte hashes preserve original file evidence."]}
    output["summary.json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return output


def check_outputs(output, expected):
    """Compare snapshots while allowing byte-only CSV/JSON serialization differences."""
    mismatches = []
    for name, data in expected.items():
        path = output / name
        if not path.exists():
            mismatches.append(name)
            continue
        actual = path.read_bytes()
        if name.endswith(".csv"):
            def parsed(blob):
                reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig"), newline=""))
                fields = reader.fieldnames
                rows = list(reader)
                if name == "file_manifest.csv":
                    rows = [{key: value for key, value in row.items() if key != "sha256_bytes"} for row in rows]
                return fields, rows
            equal = parsed(actual) == parsed(data)
        else:
            equal = json.loads(actual) == json.loads(data)
        if not equal:
            mismatches.append(name)
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproducibility/generated")
    parser.add_argument("--check", action="store_true", help="Check the output snapshot without writing; ignore byte-only serialization changes")
    args = parser.parse_args()
    output = build_outputs()
    if args.check:
        mismatches = check_outputs(args.output, output)
        if mismatches:
            parser.exit(1, "Snapshot mismatch: " + ", ".join(mismatches) + "\n")
        print("Snapshot verified (normalized source content; byte hashes may differ).")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in output.items():
            (args.output / name).write_bytes(data)
        print(f"Wrote {len(output)} inventory files to {args.output}")


if __name__ == "__main__":
    main()
