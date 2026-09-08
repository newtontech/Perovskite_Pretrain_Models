"""Export the unsplit experimental observations shared by all five archives."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {
    "CAS number": "cas_number", "Pubchem CID": "pubchem_cid",
    "Name(full)": "name_full", "Name(simple)": "name_short",
    "DOI number": "doi", "SMILES": "smiles",
    "Initial PCE": "initial_pce", "ΔPCE": "delta_pce",
    "Final PCE": "final_pce", "Date": "date", "Batch": "batch", "Note": "note",
}


def build_dataset(root=ROOT):
    reference = None
    for seed in range(5):
        records = []
        for filename in ("train_pool.csv", "test_raw.csv"):
            path = root / "baselines" / "datasets" / f"split_seed_{seed}" / filename
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not set(COLUMNS).issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing observation columns: {path}")
                for row in reader:
                    if None in row or any(row[key] is None for key in COLUMNS):
                        raise ValueError(f"Malformed observation: {path}")
                    records.append(tuple(row[key] for key in COLUMNS))
        current = Counter(records)
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"Observation multiset differs for outer seed {seed}")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(COLUMNS.values())
    # Stable lexical ordering; preserve every repeated experimental record.
    writer.writerows(sorted(reference.elements()))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build_dataset()
    destination = ROOT / "data" / "experimental_records.csv"
    if args.check:
        if destination.read_text(encoding="utf-8") != content:
            raise SystemExit("Unsplit dataset differs from the source observations")
        print("Unsplit dataset verified against all five outer split archives.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        print(f"Exported {destination}")


if __name__ == "__main__":
    main()
