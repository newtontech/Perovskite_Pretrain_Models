#!/usr/bin/env python
"""
UniMol Fine-tuning Script for Perovskite Additive Prediction
=============================================================

This script provides a command-line interface for training UniMol models
on perovskite solar cell additive data.

Usage:
    python unimol_finetune.py --data_path ../baselines/data/split_seed_0/train.csv \
                              --save_path ./output --epochs 100 --learning_rate 1e-4
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Set UniMol weights directory
os.environ['UNIMOL_WEIGHT_DIR'] = os.environ.get(
    'UNIMOL_WEIGHT_DIR',
    os.path.expanduser('~/.local/lib/python3.10/site-packages/unimol_tools/weights')
)

from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune UniMol model for perovskite additive prediction'
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data CSV file')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data CSV file (optional)')
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Name of SMILES column')
    parser.add_argument('--target_col', type=str, default='TARGET',
                        help='Name of target column')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='unimolv2',
                        choices=['unimolv1', 'unimolv2'],
                        help='UniMol model version')
    parser.add_argument('--model_size', type=str, default='84m',
                        choices=['84m', '164m', '310m', '570m', '1.1B'],
                        help='Model size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--kfold', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--max_norm', type=float, default=5.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Warmup ratio for learning rate scheduler')

    # Other arguments
    parser.add_argument('--save_path', type=str, default='./unimol_output',
                        help='Directory to save model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--remove_hs', action='store_true',
                        help='Remove hydrogens from molecules')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'scaffold'],
                        help='Data split method')

    return parser.parse_args()


def load_data(data_path, smiles_col, target_col):
    """Load and prepare data for UniMol."""
    df = pd.read_csv(data_path)

    # Check required columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in data")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Create clean DataFrame
    unimol_df = df[[smiles_col, target_col]].copy()
    unimol_df.columns = ['SMILES', 'TARGET']

    return unimol_df


def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred)[0],
        'spearman': spearmanr(y_true, y_pred)[0],
    }
    return metrics


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("UniMol Fine-tuning for Perovskite Additive Prediction")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load training data
    print(f"\nLoading training data from: {args.data_path}")
    train_df = load_data(args.data_path, args.smiles_col, args.target_col)
    print(f"Training samples: {len(train_df)}")

    # Save training data
    train_csv_path = os.path.join(args.save_path, 'train_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Load test data if provided
    test_df = None
    if args.test_path:
        print(f"Loading test data from: {args.test_path}")
        test_df = load_data(args.test_path, args.smiles_col, args.target_col)
        print(f"Test samples: {len(test_df)}")
        test_csv_path = os.path.join(args.save_path, 'test_data.csv')
        test_df.to_csv(test_csv_path, index=False)

    # Define training parameters
    params = {
        'task': 'regression',
        'data_type': 'molecule',
        'model_name': args.model_name,
        'model_size': args.model_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'early_stopping': args.early_stopping,
        'metrics': 'r2',
        'split': args.split,
        'kfold': args.kfold,
        'remove_hs': args.remove_hs,
        'random_state': args.seed,
        'target_normalize': 'auto',
        'max_norm': args.max_norm,
        'warmup_ratio': args.warmup_ratio,
    }

    print("\nTraining parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Train model
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)

    clf = MolTrain(save_path=args.save_path, **params)
    clf.fit(train_csv_path)

    print("\nTraining completed!")

    # Make predictions
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    predictor = MolPredict(load_model=args.save_path)

    # Training predictions
    train_pred = predictor.predict(train_csv_path)
    train_metrics = evaluate_predictions(train_df['TARGET'].values, train_pred)

    print("\nTraining Set Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test predictions (if test data provided)
    if test_df is not None:
        test_pred = predictor.predict(test_csv_path)
        test_metrics = evaluate_predictions(test_df['TARGET'].values, test_pred)

        print("\nTest Set Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        'train_metrics': train_metrics,
        'params': params,
        'timestamp': datetime.now().isoformat(),
    }

    if test_df is not None:
        results['test_metrics'] = test_metrics

    with open(os.path.join(args.save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.save_path}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    main()
