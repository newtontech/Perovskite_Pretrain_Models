"""Small deterministic property-prediction helpers.

These utilities provide a testable scaffold for issues #5 and #6 without
depending on heavy ML frameworks or trained artifacts. They are intentionally
simple and should be replaced by calibrated model adapters for publication
metrics.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


ATOM_PATTERN = re.compile(r"Cl|Br|[A-Z][a-z]?|[cnops]")
DEFAULT_TARGET_COLUMNS = {
    "delta_pce": "TARGET",
    "bandgap": "BANDGAP_EV",
    "stability_t80": "T80_HOURS",
}


@dataclass(frozen=True)
class PropertyRow:
    """One molecule with whichever property labels are available."""

    smiles: str
    targets: dict[str, float]


@dataclass(frozen=True)
class RegressionMetrics:
    """Regression metrics for one target on one evaluated split."""

    n: int
    mae: float
    rmse: float
    r2: float | None


@dataclass(frozen=True)
class TaskBenchmark:
    """Benchmark summary for one property target."""

    task: str
    metrics: RegressionMetrics
    acceptance_mae: float | None
    acceptance_passed: bool | None
    top_features: list[tuple[str, float]]


def load_property_rows(
    csv_path: str | Path,
    smiles_column: str = "SMILES",
    target_columns: Mapping[str, str] | None = None,
) -> list[PropertyRow]:
    """Load property rows while skipping missing target values per task."""

    path = Path(csv_path)
    columns = dict(target_columns or DEFAULT_TARGET_COLUMNS)
    rows: list[PropertyRow] = []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            smiles = (raw.get(smiles_column) or "").strip()
            if not smiles:
                continue

            targets: dict[str, float] = {}
            for task_name, column_name in columns.items():
                value = (raw.get(column_name) or "").strip()
                if value:
                    targets[task_name] = float(value)
            if targets:
                rows.append(PropertyRow(smiles=smiles, targets=targets))

    return rows


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> RegressionMetrics:
    """Compute dependency-free regression metrics."""

    actual_values = list(actual)
    predicted_values = list(predicted)
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    if not actual_values:
        raise ValueError("At least one prediction is required")

    errors = [prediction - truth for truth, prediction in zip(actual_values, predicted_values, strict=True)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))

    mean_actual = sum(actual_values) / len(actual_values)
    total_sum_squares = sum((truth - mean_actual) ** 2 for truth in actual_values)
    if total_sum_squares == 0:
        r2 = None
    else:
        residual_sum_squares = sum(error * error for error in errors)
        r2 = 1.0 - (residual_sum_squares / total_sum_squares)

    return RegressionMetrics(n=len(actual_values), mae=mae, rmse=rmse, r2=r2)


def leave_one_out_benchmark(
    rows: Iterable[PropertyRow],
    k_neighbors: int = 3,
    acceptance_mae: Mapping[str, float] | None = None,
    top_n_features: int = 5,
) -> dict[str, TaskBenchmark]:
    """Evaluate the scaffold predictor with deterministic leave-one-out splits."""

    row_list = list(rows)
    if len(row_list) < 2:
        raise ValueError("At least two labeled rows are required for leave-one-out evaluation")

    task_names = sorted({task for row in row_list for task in row.targets})
    thresholds = dict(acceptance_mae or {})
    reports: dict[str, TaskBenchmark] = {}

    for task in task_names:
        actual: list[float] = []
        predicted: list[float] = []
        labeled_indexes = [index for index, row in enumerate(row_list) if task in row.targets]
        if len(labeled_indexes) < 2:
            continue

        for held_out_index in labeled_indexes:
            train_rows = [row for index, row in enumerate(row_list) if index != held_out_index]
            predictor = DeterministicPropertyPredictor(k_neighbors=k_neighbors).fit(train_rows)
            prediction = predictor.predict_smiles(row_list[held_out_index].smiles)
            actual.append(row_list[held_out_index].targets[task])
            predicted.append(prediction[task])

        fitted_predictor = DeterministicPropertyPredictor(k_neighbors=k_neighbors).fit(row_list)
        importance = list(fitted_predictor.feature_importance(task).items())[:top_n_features]
        metrics = regression_metrics(actual, predicted)
        threshold = thresholds.get(task)
        reports[task] = TaskBenchmark(
            task=task,
            metrics=metrics,
            acceptance_mae=threshold,
            acceptance_passed=None if threshold is None else metrics.mae < threshold,
            top_features=importance,
        )

    return reports


def write_benchmark_report(
    report_path: str | Path,
    benchmarks: Mapping[str, TaskBenchmark],
    dataset_manifest: Mapping[str, str] | None = None,
) -> None:
    """Write a portable JSON report for property-prediction benchmark runs."""

    payload = {
        "dataset_manifest": dict(dataset_manifest or {}),
        "benchmarks": {task: asdict(report) for task, report in sorted(benchmarks.items())},
    }
    Path(report_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smiles_features(smiles: str) -> dict[str, float]:
    """Return compact deterministic features from a SMILES string."""

    atoms = [match.group(0).capitalize() for match in ATOM_PATTERN.finditer(smiles)]
    features: dict[str, float] = {
        "length": float(len(smiles)),
        "branch_count": float(smiles.count("(")),
        "ring_digit_count": float(sum(char.isdigit() for char in smiles)),
        "double_bond_count": float(smiles.count("=")),
        "triple_bond_count": float(smiles.count("#")),
        "aromatic_count": float(sum(char in "cnops" for char in smiles)),
    }

    for atom in atoms:
        features[f"atom_{atom}"] = features.get(f"atom_{atom}", 0.0) + 1.0
    return features


def _distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return math.sqrt(sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys))


class DeterministicPropertyPredictor:
    """Deterministic k-nearest-neighbor regressor for scaffold workflows."""

    def __init__(self, k_neighbors: int = 3):
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        self.k_neighbors = k_neighbors
        self._rows: list[PropertyRow] = []
        self._features: list[dict[str, float]] = []

    @property
    def tasks(self) -> list[str]:
        task_names = {task for row in self._rows for task in row.targets}
        return sorted(task_names)

    def fit(self, rows: Iterable[PropertyRow]) -> "DeterministicPropertyPredictor":
        self._rows = list(rows)
        if not self._rows:
            raise ValueError("At least one labeled row is required")
        self._features = [smiles_features(row.smiles) for row in self._rows]
        return self

    def fit_smiles(
        self, rows: Iterable[tuple[str, Mapping[str, float]]]
    ) -> "DeterministicPropertyPredictor":
        return self.fit(PropertyRow(smiles, dict(targets)) for smiles, targets in rows)

    def predict_smiles(self, smiles: str) -> dict[str, float]:
        if not self._rows:
            raise ValueError("Predictor must be fitted before prediction")

        query_features = smiles_features(smiles)
        predictions: dict[str, float] = {}
        for task in self.tasks:
            neighbors = []
            for row, features in zip(self._rows, self._features, strict=True):
                if task in row.targets:
                    neighbors.append((_distance(query_features, features), row.targets[task]))
            neighbors.sort(key=lambda item: (item[0], item[1]))
            selected = neighbors[: self.k_neighbors]
            weights = [1.0 / (distance + 1.0) for distance, _ in selected]
            total_weight = sum(weights)
            predictions[task] = sum(
                weight * value for weight, (_, value) in zip(weights, selected, strict=True)
            ) / total_weight

        return predictions

    def feature_importance(self, task: str) -> dict[str, float]:
        """Estimate task sensitivity via absolute feature-target covariance."""

        labeled = [
            (features, row.targets[task])
            for row, features in zip(self._rows, self._features, strict=True)
            if task in row.targets
        ]
        if not labeled:
            raise ValueError(f"No fitted rows have target {task!r}")

        mean_target = sum(target for _, target in labeled) / len(labeled)
        keys = sorted({key for features, _ in labeled for key in features})
        scores = {}
        for key in keys:
            mean_feature = sum(features.get(key, 0.0) for features, _ in labeled) / len(labeled)
            covariance = sum(
                (features.get(key, 0.0) - mean_feature) * (target - mean_target)
                for features, target in labeled
            )
            scores[key] = abs(covariance) / len(labeled)
        return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
