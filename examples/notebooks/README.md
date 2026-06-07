# Perovskite Property Prediction Examples

This directory contains runnable examples demonstrating the perovskite property prediction and generation workflows.

## Quick Start

Run the complete demo:
```bash
python3 property_prediction_demo.py
```

## Examples

### Property Prediction Demo (`property_prediction_demo.py`)

Demonstrates three core capabilities:

1. **Property Prediction**: Train a simple predictor and make predictions for new perovskite molecules
2. **Feature Importance**: Understand which molecular features influence predictions
3. **Benchmark Evaluation**: Run leave-one-out cross-validation to estimate model performance

### Output

The demo shows:
- Predicted PCE (Power Conversion Efficiency) and bandgap values
- Top molecular features affecting predictions
- Cross-validation metrics (MAE, RMSE, R²) for multi-task prediction
- Molecular generation and screening workflow

## Usage for Real Workflows

Replace synthetic data with real perovskite datasets:

```python
import pandas as pd
from perovskite_pretrain.property_prediction import load_property_rows, leave_one_out_benchmark

# Load your data
df = pd.read_csv("path/to/perovskite_data.csv")
df.to_csv("train.csv", index=False)

# Run benchmark
rows = load_property_rows("train.csv")
benchmarks = leave_one_out_benchmark(
    rows,
    k_neighbors=3,
    acceptance_mae={"delta_pce": 2.0, "bandgap": 0.1}
)
```

## Next Steps

For full model training:
- Use `train/run.py` for Uni-Mol fine-tuning
- Use `train/train_molclr/finetune.py` for MolCLR
- Run baseline comparisons in `baselines/baseline_search_get.py`

## Requirements

The demo uses only standard library and the project's own modules (no heavy ML dependencies).
For production workflows with pre-trained models, see `requirements.txt`.
