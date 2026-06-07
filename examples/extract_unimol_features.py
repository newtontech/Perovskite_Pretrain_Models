"""Extract Uni-Mol prediction features for reproducible comparison runs.

This example keeps heavy imports inside ``main`` so static CI can compile the
file without installing ML dependencies. Generated feature tensors should be
written under an ignored run directory, not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Uni-Mol features from a trained model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing Uni-Mol config.yaml and fold checkpoints.")
    parser.add_argument("--input-csv", required=True, help="CSV containing the configured SMILES and target columns.")
    parser.add_argument("--output-path", required=True, help="Path for the generated torch feature tensor.")
    parser.add_argument("--random-weight", action="store_true", help="Reset model weights before prediction as a control.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for random-weight initialization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_path = repo_root / "train"
    sys.path.insert(0, str(train_path))

    import torch
    from unimol_tools import MolPredict

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = MolPredict(
        load_model=args.model_dir,
        random_weight=args.random_weight,
        random_seed=args.seed,
    )
    _, features = predictor.predict(args.input_csv)
    torch.save(features[0], output_path)
    print(f"Saved features with shape {tuple(features[0].shape)} to {output_path}")


if __name__ == "__main__":
    main()
