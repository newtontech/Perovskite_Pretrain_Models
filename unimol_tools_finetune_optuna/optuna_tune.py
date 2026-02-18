#!/usr/bin/env python
"""
Optuna Hyperparameter Optimization for UniMol
==============================================

This script performs hyperparameter optimization using Optuna to find
the best configuration for UniMol fine-tuning on perovskite data.

Usage:
    python optuna_tune.py --data_path ../baselines/data/split_seed_0/train.csv \
                          --n_trials 100 --storage sqlite:///study.db
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil

# Set UniMol weights directory
os.environ['UNIMOL_WEIGHT_DIR'] = os.environ.get(
    'UNIMOL_WEIGHT_DIR',
    os.path.expanduser('~/.local/lib/python3.10/site-packages/unimol_tools/weights')
)

from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import r2_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter optimization for UniMol'
    )

    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data CSV file')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data CSV file (optional)')

    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                        help='Optuna storage URL')
    parser.add_argument('--study_name', type=str, default='unimol_perovskite',
                        help='Name of the study')
    parser.add_argument('--save_dir', type=str, default='./optuna_trials',
                        help='Directory to save trial models')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


class UniMolOptimizer:
    """Optimizer class for UniMol hyperparameter tuning."""

    def __init__(self, train_df, test_df, save_dir, seed=42):
        """
        Initialize optimizer.

        Parameters:
            train_df: Training DataFrame with SMILES and TARGET columns
            test_df: Test DataFrame (optional, for validation)
            save_dir: Base directory for saving trial results
            seed: Random seed
        """
        self.train_df = train_df
        self.test_df = test_df
        self.save_dir = save_dir
        self.seed = seed

        os.makedirs(save_dir, exist_ok=True)

    def objective(self, trial):
        """
        Optuna objective function.

        Parameters:
            trial: Optuna trial object

        Returns:
            float: Validation R² score (to maximize)
        """
        # Define search space
        params = {
            'task': 'regression',
            'data_type': 'molecule',
            'model_name': trial.suggest_categorical('model_name', ['unimolv1', 'unimolv2']),
            'model_size': trial.suggest_categorical('model_size', ['84m', '164m', '310m']),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 20, 150),
            'early_stopping': trial.suggest_int('early_stopping', 5, 30),
            'kfold': trial.suggest_categorical('kfold', [3, 5]),
            'max_norm': trial.suggest_float('max_norm', 1.0, 10.0),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.1),
            'remove_hs': trial.suggest_categorical('remove_hs', [True, False]),
            'split': 'random',
            'metrics': 'r2',
            'random_state': self.seed,
            'target_normalize': trial.suggest_categorical('target_normalize',
                                                          ['auto', 'standard', 'minmax']),
        }

        # Create trial directory
        trial_dir = os.path.join(self.save_dir, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)

        # Save training data
        train_csv = os.path.join(trial_dir, 'train.csv')
        self.train_df.to_csv(train_csv, index=False)

        try:
            # Train model
            clf = MolTrain(save_path=trial_dir, **params)
            clf.fit(train_csv)

            # Evaluate on test set if available
            if self.test_df is not None:
                test_csv = os.path.join(trial_dir, 'test.csv')
                self.test_df.to_csv(test_csv, index=False)

                predictor = MolPredict(load_model=trial_dir)
                predictions = predictor.predict(test_csv)
                score = r2_score(self.test_df['TARGET'].values, predictions)
            else:
                # Use training predictions as proxy
                predictor = MolPredict(load_model=trial_dir)
                predictions = predictor.predict(train_csv)
                score = r2_score(self.train_df['TARGET'].values, predictions)

            # Save trial params and score
            trial_info = {
                'trial_number': trial.number,
                'params': params,
                'score': score,
            }
            with open(os.path.join(trial_dir, 'trial_info.json'), 'w') as f:
                json.dump(trial_info, f, indent=2)

            return score

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -1.0

    def run_optimization(self, n_trials=100, n_jobs=1, storage=None, study_name=None):
        """
        Run Optuna optimization.

        Parameters:
            n_trials: Number of trials
            n_jobs: Number of parallel jobs
            storage: Optuna storage URL
            study_name: Name of the study

        Returns:
            optuna.Study: Completed study
        """
        study = optuna.create_study(
            study_name=study_name or 'unimol_optimization',
            storage=storage,
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(),
            load_if_exists=True,
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        return study


def load_data(data_path, smiles_col='SMILES', target_col='TARGET'):
    """Load and prepare data."""
    df = pd.read_csv(data_path)

    unimol_df = df[[smiles_col, target_col]].copy()
    unimol_df.columns = ['SMILES', 'TARGET']

    return unimol_df


def main():
    """Main function."""
    args = parse_args()

    print("=" * 60)
    print("Optuna Hyperparameter Optimization for UniMol")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    train_df = load_data(args.data_path)

    test_df = None
    if args.test_path:
        print(f"Loading test data from: {args.test_path}")
        test_df = load_data(args.test_path)

    print(f"Training samples: {len(train_df)}")
    if test_df is not None:
        print(f"Test samples: {len(test_df)}")

    # Create optimizer
    optimizer = UniMolOptimizer(
        train_df=train_df,
        test_df=test_df,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    # Run optimization
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study = optimizer.run_optimization(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best R² score: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params_path = os.path.join(args.save_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)

    print(f"\nBest parameters saved to: {best_params_path}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print dashboard command
    print(f"\nTo view optimization dashboard, run:")
    print(f"  optuna-dashboard {args.storage}")

    return study


if __name__ == '__main__':
    main()
