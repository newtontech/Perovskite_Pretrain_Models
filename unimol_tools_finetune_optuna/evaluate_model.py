#!/usr/bin/env python
"""
Model Evaluation Utilities
==========================

This script provides utilities for evaluating trained UniMol models
and generating comparison reports.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Set UniMol weights directory
os.environ['UNIMOL_WEIGHT_DIR'] = os.environ.get(
    'UNIMOL_WEIGHT_DIR',
    os.path.expanduser('~/.local/lib/python3.10/site-packages/unimol_tools/weights')
)

from unimol_tools import MolPredict


def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred)[0],
        'spearman': spearmanr(y_true, y_pred)[0],
    }


def plot_predictions(y_true, y_pred, title='Predictions vs Actual', save_path=None):
    """Create scatter plot of predictions vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    # Add metrics text
    metrics = calculate_metrics(y_true, y_pred)
    text = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nPearson = {metrics['pearson']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_residuals(y_true, y_pred, title='Residuals Distribution', save_path=None):
    """Create residual plots."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')

    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def evaluate_model(model_path, test_csv, output_dir='./evaluation'):
    """
    Evaluate a trained model on test data.

    Parameters:
        model_path: Path to trained model directory
        test_csv: Path to test data CSV
        output_dir: Directory to save evaluation results

    Returns:
        dict: Evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df)} test samples")

    # Load model
    predictor = MolPredict(load_model=model_path)

    # Make predictions
    predictions = predictor.predict(test_csv)

    # Calculate metrics
    metrics = calculate_metrics(test_df['TARGET'].values, predictions)

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Generate plots
    plot_predictions(
        test_df['TARGET'].values, predictions,
        title='Test Set Predictions',
        save_path=os.path.join(output_dir, 'predictions.png')
    )

    plot_residuals(
        test_df['TARGET'].values, predictions,
        title='Test Set Residuals',
        save_path=os.path.join(output_dir, 'residuals.png')
    )

    # Save results
    results = {
        'metrics': metrics,
        'model_path': model_path,
        'test_csv': test_csv,
        'n_samples': len(test_df),
    }

    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'SMILES': test_df['SMILES'],
        'actual': test_df['TARGET'],
        'predicted': predictions,
        'residual': test_df['TARGET'] - predictions,
    })
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"\nResults saved to: {output_dir}")

    return metrics


def compare_models(model_results, output_path='./model_comparison.png'):
    """
    Create comparison plot for multiple models.

    Parameters:
        model_results: Dictionary of {model_name: metrics_dict}
        output_path: Path to save comparison plot
    """
    metrics_names = ['r2', 'rmse', 'pearson', 'spearman']
    model_names = list(model_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_names):
        values = [model_results[name].get(metric, 0) for name in model_names]

        axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.upper()} Comparison', fontsize=12)
        axes[i].set_ylabel(metric.upper())

        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Evaluate UniMol model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for results')

    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_csv, args.output_dir)


if __name__ == '__main__':
    main()
