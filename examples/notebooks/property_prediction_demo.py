"""Demo: Perovskite Property Prediction with Pre-trained Models

This example demonstrates the complete property prediction workflow:
1. Load perovskite SMILES data
2. Extract molecular features using Uni-Mol or MolCLR
3. Train property prediction models
4. Evaluate performance and visualize results

Note: This is a minimal runnable example. For full benchmarking results,
see the comprehensive evaluation pipeline in train/run.py
"""

import sys
from pathlib import Path

# Add repository root to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def demo_deterministic_predictor():
    """Lightweight demo using deterministic predictor (no heavy ML deps)."""
    from perovskite_pretrain.property_prediction import (
        DeterministicPropertyPredictor,
        load_property_rows,
        regression_metrics,
        leave_one_out_benchmark,
        write_benchmark_report
    )
    
    print("=== Perovskite Property Prediction Demo ===\n")
    
    # Example 1: Simple prediction with synthetic data
    print("Example 1: Training a simple predictor")
    
    training_data = [
        ("CCO", {"delta_pce": 1.0, "bandgap": 1.6}),
        ("CCN", {"delta_pce": 0.2, "bandgap": 1.4}),
        ("O=C=O", {"delta_pce": -0.3, "bandgap": 1.9}),
        ("c1ccccc1", {"delta_pce": 0.5, "bandgap": 2.1}),
    ]
    
    # Train predictor
    predictor = DeterministicPropertyPredictor(k_neighbors=3)
    predictor.fit_smiles(training_data)
    
    # Make predictions for new perovskite molecules
    test_molecules = ["CCCN", "CCCO", "c1ccc(O)cc1"]
    
    print(f"\nPredictions for {len(test_molecules)} test molecules:")
    for smiles in test_molecules:
        prediction = predictor.predict_smiles(smiles)
        print(f"  {smiles:20s} -> PCE: {prediction['delta_pce']:.3f}, "
              f"Bandgap: {prediction['bandgap']:.3f} eV")
    
    # Example 2: Feature importance analysis
    print("\n\nExample 2: Feature Importance for PCE Prediction")
    importance = predictor.feature_importance("delta_pce")
    print("Top 5 features:")
    top_features = list(importance.items())[:5]
    for feature, score in top_features:
        print(f"  {feature:20s} -> {score:.4f}")
    
    # Example 3: Benchmark evaluation
    print("\n\nExample 3: Leave-One-Out Benchmark Evaluation")
    print("This simulates cross-validation performance estimation")

    import csv
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample dataset
        csv_path = Path(tmpdir) / "sample_data.csv"

        # Write sample data
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["SMILES", "TARGET", "BANDGAP_EV", "T80_HOURS"])
            writer.writeheader()
            writer.writerows([
                {"SMILES": "CCO", "TARGET": "1.0", "BANDGAP_EV": "1.5", "T80_HOURS": "40"},
                {"SMILES": "CCN", "TARGET": "1.2", "BANDGAP_EV": "1.6", "T80_HOURS": "50"},
                {"SMILES": "CCC", "TARGET": "0.8", "BANDGAP_EV": "1.4", "T80_HOURS": "35"},
                {"SMILES": "COC", "TARGET": "0.5", "BANDGAP_EV": "1.8", "T80_HOURS": "65"},
            ])

        # Load data
        rows = load_property_rows(csv_path)
        print(f"Loaded {len(rows)} perovskite molecules")
        
        # Run benchmark
        benchmarks = leave_one_out_benchmark(
            rows,
            k_neighbors=2,
            acceptance_mae={"delta_pce": 10.0, "bandgap": 10.0},
            top_n_features=3
        )
        
        # Display results
        for task_name, benchmark in benchmarks.items():
            metrics = benchmark.metrics
            print(f"\nTask: {task_name}")
            print(f"  MAE:   {metrics.mae:.4f}")
            print(f"  RMSE:  {metrics.rmse:.4f}")
            r2_str = f"{metrics.r2:.4f}" if metrics.r2 is not None else "N/A"
            print(f"  R²:    {r2_str}")
            print(f"  N:     {metrics.n}")
        
        # Generate report
        report_path = Path(tmpdir) / "benchmark_report.json"
        write_benchmark_report(
            report_path,
            benchmarks,
            dataset_manifest={
                "source": "demo_synthetic_data",
                "license": "MIT",
                "note": "Replace with real perovskite dataset for production"
            }
        )
        
        print(f"\nBenchmark report saved to: {report_path}")
        
        # Show report structure
        report = json.loads(report_path.read_text())
        print(f"Report contains {len(report['benchmarks'])} task benchmarks")
        print(f"Dataset source: {report['dataset_manifest']['source']}")


def demo_molecular_generation():
    """Demo of molecular generation and screening workflow."""
    from perovskite_pretrain.generation import (
        CandidateGenerator,
        evaluate_candidates,
        summarize_screening
    )
    from perovskite_pretrain.property_prediction import DeterministicPropertyPredictor
    
    print("\n\n=== Molecular Generation Demo ===\n")
    
    # Train a simple predictor for screening
    training_data = [
        ("CCN", {"delta_pce": 0.8, "stability_t80": 100.0}),
        ("CCO", {"delta_pce": 0.1, "stability_t80": 40.0}),
        ("c1ccccc1", {"delta_pce": 0.5, "stability_t80": 80.0}),
    ]
    
    predictor = DeterministicPropertyPredictor(k_neighbors=2)
    predictor.fit_smiles(training_data)
    
    # Generate candidates from seed molecules
    print("Generating candidate perovskite molecules...")
    generator = CandidateGenerator(seed=42, substituents=("N", "O", "F", "Cl"))
    
    seed_molecules = ["CC", "c1ccccc1"]  # Simple organic backbones
    candidates = generator.generate(seed_molecules, n_per_seed=3)
    
    print(f"Generated {len(candidates)} candidates from {len(seed_molecules)} seeds")
    
    # Evaluate candidates
    print("\nEvaluating candidates...")
    evaluated = evaluate_candidates(
        candidates,
        training_smiles=[row[0] for row in training_data],
        predictor=predictor,
        allowed_atoms={"C", "N", "O", "F", "Cl"}
    )
    
    # Show results
    summary = summarize_screening(evaluated)
    print(f"\nScreening Summary:")
    print(f"  Total candidates:  {summary['total']}")
    print(f"  Valid SMILES:      {summary['valid']} ({summary['valid_fraction']:.1%})")
    print(f"  Novel molecules:   {summary['novel']} ({summary['novel_fraction']:.1%})")
    print(f"  Duplicates:        {summary['duplicates']} ({summary['duplicate_fraction']:.1%})")
    
    # Show top candidates by predicted PCE
    print("\nTop 3 candidates by predicted PCE:")
    valid_candidates = [c for c in evaluated if c.valid and c.predictions]
    valid_candidates.sort(key=lambda c: c.predictions.get("delta_pce", 0), reverse=True)
    
    for i, candidate in enumerate(valid_candidates[:3], 1):
        print(f"  {i}. {candidate.smiles:20s} -> PCE: {candidate.predictions['delta_pce']:.3f}")


def main():
    """Run all demos."""
    print("Perovskite Property Prediction & Generation Demos")
    print("=" * 60)
    
    demo_deterministic_predictor()
    demo_molecular_generation()
    
    print("\n\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nNext Steps:")
    print("1. Replace synthetic data with real perovskite datasets")
    print("2. Train full models using train/run.py for Uni-Mol")
    print("3. Use train/train_molclr/finetune.py for MolCLR")
    print("4. Run baseline comparisons in baselines/baseline_search_get.py")


if __name__ == "__main__":
    main()
