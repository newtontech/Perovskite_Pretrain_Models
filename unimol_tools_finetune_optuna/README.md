# UniMol Fine-tuning with Optuna Optimization

This folder contains a complete workflow for fine-tuning UniMol models on the perovskite additive dataset with hyperparameter optimization using Optuna.

## Overview

The workflow includes:
1. Data preparation from the Perovskite_Pretrain_Models dataset
2. UniMol model fine-tuning using the unimol_tools library
3. Hyperparameter optimization with Optuna
4. Model evaluation and comparison

## Files

- `unimol_finetune_optuna.ipynb` - Complete Jupyter notebook with the workflow
- `unimol_finetune.py` - Python script for training without notebook
- `optuna_tune.py` - Optuna hyperparameter optimization script
- `run_slurm.sh` - SLURM batch script for cluster execution
- `evaluate_model.py` - Model evaluation utilities

## Requirements

```bash
# Install UniMol tools
pip install unimol-tools

# Or use Docker
docker pull dptechnology/unimol_tools:latest

# Additional dependencies
pip install optuna optuna-dashboard pandas scikit-learn matplotlib
```

## Data

The data splits are located at:
- `../baselines/data/split_seed_0/train.csv` through `split_seed_4/`
- Each split contains:
  - `train.csv`: Training set (~1,626 samples)
  - `test.csv`: Test set (~377 samples)
  - Columns: `SMILES`, `TARGET`, plus 22 DFT features

## Quick Start

### Option 1: Jupyter Notebook

```bash
jupyter notebook unimol_finetune_optuna.ipynb
```

### Option 2: Python Script

```bash
python unimol_finetune.py --data_path ../baselines/data/split_seed_0/train.csv \
                          --save_path ./output \
                          --epochs 100 \
                          --learning_rate 1e-4
```

### Option 3: SLURM Cluster

```bash
sbatch run_slurm.sh
```

### Option 4: Optuna Optimization

```bash
python optuna_tune.py --n_trials 100

# Monitor with dashboard
optuna-dashboard sqlite:///optuna_study.db
```

## Hyperparameter Search Space

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| model_name | ['unimolv1', 'unimolv2'] | 'unimolv2' |
| model_size | ['84m', '164m', '310m', '570m'] | '84m' |
| learning_rate | loguniform(1e-6, 1e-3) | 1e-4 |
| batch_size | [4, 8, 16, 32, 64] | 16 |
| epochs | [10, 200] | 100 |
| early_stopping | [5, 50] | 20 |
| kfold | [5, 10] | 5 |
| max_norm | uniform(1.0, 10.0) | 5.0 |
| warmup_ratio | uniform(0.0, 0.1) | 0.03 |
| remove_hs | [True, False] | False |
| split | ['random', 'scaffold'] | 'random' |
| target_normalize | ['auto', 'none', 'minmax', 'standard', 'robust'] | 'auto' |

## Expected Results

Based on previous experiments:
- **R² score**: 0.45 - 0.65
- **RMSE**: 0.8 - 1.2 (for delta_PCE)
- **Best model**: UniMol v2, 84m size, learning rate ~1e-4

## References

1. UniMol: https://github.com/dptech-corp/Uni-Mol
2. UniMol Tools: https://github.com/dptech-corp/unimol-tools
3. Optuna: https://optuna.readthedocs.io/
