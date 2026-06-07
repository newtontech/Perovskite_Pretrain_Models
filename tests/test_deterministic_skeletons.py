import csv
import json
from pathlib import Path

from perovskite_pretrain.generation import (
    CandidateGenerator,
    evaluate_candidates,
    summarize_screening,
)
from perovskite_pretrain.property_prediction import (
    DeterministicPropertyPredictor,
    leave_one_out_benchmark,
    load_property_rows,
    regression_metrics,
    write_benchmark_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_property_fixture(path):
    return write_csv(
        path,
        [
            {"SMILES": "CCO", "TARGET": "1.0", "BANDGAP_EV": "1.5", "T80_HOURS": "40"},
            {"SMILES": "CCN", "TARGET": "1.2", "BANDGAP_EV": "1.6", "T80_HOURS": "50"},
            {"SMILES": "CCC", "TARGET": "0.8", "BANDGAP_EV": "1.4", "T80_HOURS": "35"},
            {"SMILES": "COC", "TARGET": "0.5", "BANDGAP_EV": "1.8", "T80_HOURS": "65"},
        ],
    )


def test_property_rows_load_only_available_targets(tmp_path):
    data_path = tmp_path / "properties.csv"
    write_csv(
        data_path,
        [
            {"SMILES": "CCO", "TARGET": "0.5", "BANDGAP_EV": "", "T80_HOURS": "10"},
            {"SMILES": "CN", "TARGET": "-0.25", "BANDGAP_EV": "1.62", "T80_HOURS": ""},
        ],
    )

    rows = load_property_rows(
        data_path,
        target_columns={
            "delta_pce": "TARGET",
            "bandgap": "BANDGAP_EV",
            "stability_t80": "T80_HOURS",
        },
    )

    assert rows[0].targets == {"delta_pce": 0.5, "stability_t80": 10.0}
    assert rows[1].targets == {"delta_pce": -0.25, "bandgap": 1.62}


def test_property_predictor_is_deterministic_and_reports_importance():
    rows = [
        ("CCO", {"delta_pce": 1.0, "bandgap": 1.6}),
        ("CN", {"delta_pce": 0.2, "bandgap": 1.4}),
        ("O=C=O", {"delta_pce": -0.3, "bandgap": 1.9}),
    ]
    predictor = DeterministicPropertyPredictor(k_neighbors=2)
    predictor.fit_smiles(rows)

    first = predictor.predict_smiles("CCN")
    second = predictor.predict_smiles("CCN")

    assert first == second
    assert set(first) == {"delta_pce", "bandgap"}
    assert first["delta_pce"] > 0.0
    assert predictor.feature_importance("delta_pce")["atom_C"] > 0


def test_property_benchmark_reports_metrics_and_acceptance(tmp_path):
    rows = load_property_rows(write_property_fixture(tmp_path / "properties.csv"))

    metrics = regression_metrics([1.0, 2.0, 3.0], [1.2, 2.2, 2.6])
    assert metrics.n == 3
    assert round(metrics.mae, 6) == round((0.2 + 0.2 + 0.4) / 3, 6)
    assert metrics.rmse > metrics.mae
    assert metrics.r2 is not None

    benchmarks = leave_one_out_benchmark(
        rows,
        k_neighbors=1,
        acceptance_mae={"delta_pce": 10.0, "bandgap": 10.0},
        top_n_features=3,
    )
    assert {"delta_pce", "bandgap"} <= set(benchmarks)
    assert benchmarks["delta_pce"].metrics.n == 4
    assert benchmarks["delta_pce"].acceptance_passed is True
    assert benchmarks["delta_pce"].top_features

    report_path = tmp_path / "benchmark-report.json"
    write_benchmark_report(
        report_path,
        benchmarks,
        dataset_manifest={"source": "unit-test fixture", "license": "synthetic"},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset_manifest"]["source"] == "unit-test fixture"
    assert report["benchmarks"]["bandgap"]["metrics"]["n"] == 4


def test_generation_screening_scores_valid_novel_candidates():
    predictor = DeterministicPropertyPredictor(k_neighbors=1)
    predictor.fit_smiles(
        [
            ("CCN", {"delta_pce": 0.8, "stability_t80": 100.0}),
            ("CCO", {"delta_pce": 0.1, "stability_t80": 40.0}),
        ]
    )
    generator = CandidateGenerator(seed=7, substituents=("N", "O"))

    candidates = generator.generate(["CC"], n_per_seed=3)
    evaluated = evaluate_candidates(
        candidates,
        training_smiles={"CCO"},
        predictor=predictor,
        allowed_atoms={"C", "N", "O"},
    )
    summary = summarize_screening(evaluated)

    assert [candidate.smiles for candidate in candidates] == ["CCN", "CCO"]
    assert evaluated[0].valid is True
    assert evaluated[0].novel is True
    assert evaluated[1].novel is False
    assert evaluated[0].predictions["delta_pce"] == 0.8
    assert summary["total"] == 2
    assert summary["valid_fraction"] == 1.0
    assert summary["novel_fraction"] == 0.5


def test_chemberta2_config_has_comparison_metadata():
    config_path = REPO_ROOT / "configs" / "pretraining" / "chemberta2_mlm.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["comparison"]["shared_split"] == "same rows and split seeds as property_prediction_multitask"
    assert {"name", "training_mode", "downstream_protocol", "artifact_policy"} <= set(
        config["comparison"]["runs"][0]
    )
    assert "random_weight_unimol_features" in {
        run["name"] for run in config["comparison"]["runs"]
    }
