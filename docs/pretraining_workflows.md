# Perovskite Pretraining Workflows

This document turns issues #3, #5, and #6 into reproducible first-pass workflows without adding large datasets, checkpoints, logs, or generated figures to Git.

## Issue #3: ChemBERTa2 Training Detail

ChemBERTa2 experiments should be recorded as masked-language-modeling pretraining plus downstream property evaluation. The minimum run record is:

- Dataset manifest: source, license, molecule count, canonicalization rules, train/validation split, and any filtering.
- Model manifest: base checkpoint, tokenizer, max sequence length, MLM probability, seed, and full training config.
- Training metrics: validation loss and perplexity by epoch.
- Downstream comparison: frozen embeddings, fine-tuned ChemBERTa2, Uni-Mol, MolCLR, KRFP, and DFT baselines on the same split.
- Artifact references: checkpoint checksum and storage URI, never raw weights in the repository.

Config stub: `configs/pretraining/chemberta2_mlm.json`.

Example command shape:

```bash
python -m transformers.examples.pytorch.language-modeling.run_mlm \
  --model_name_or_path DeepChem/ChemBERTa-77M-MLM \
  --train_file data/chemberta2/train_smiles.txt \
  --validation_file data/chemberta2/validation_smiles.txt \
  --max_seq_length 256 \
  --mlm_probability 0.15 \
  --output_dir runs/chemberta2_mlm_seed42
```

The exact command may differ by the installed `transformers` version; preserve the resolved CLI, package versions, and config JSON with the run.

## Issue #5: Perovskite Property Prediction

The first supported property task is the existing `TARGET` residual for delta PCE in `train/train.csv`. Bandgap and stability are defined as schema stubs until public or private columns are added.

Config stub: `configs/pretraining/property_prediction_multitask.json`.

Recommended evaluation sequence:

1. Run DFT/KRFP baselines with the same split seeds.
2. Fine-tune Uni-Mol with `train/run.py`.
3. Fine-tune MolCLR with `train/train_molclr/finetune.py`.
4. Compare against ChemBERTa2 embeddings on identical train/test rows.
5. Report mean and standard deviation across seeds for MAE, RMSE, and R2.

Required report fields:

| Field | Purpose |
| --- | --- |
| target | Property name and unit. |
| split_seed | Reproducible split identifier. |
| feature_set | DFT, KRFP, Uni-Mol, MolCLR, ChemBERTa2, or random-weight control. |
| metric | MAE/RMSE/R2 with target units. |
| artifact | Local path or external URI plus checksum for weights and outputs. |

## Issue #6: Perovskite Molecular Generation

Generation work should start as a screened candidate pipeline before training large models. The first pass is:

1. Define seed molecules and allowed chemistry.
2. Generate candidates with a VAE or conditional model outside Git-tracked artifact paths.
3. Validate generated SMILES with RDKit.
4. Remove duplicates and training-set matches.
5. Score candidates with the property predictor from issue #5.
6. Commit only reviewed candidate schemas, small top-k tables, and configs.

Config stub: `configs/pretraining/molecule_generation_vae.json`.

Acceptance metrics should be reported only when the predictor is calibrated for the target property. Until then, record validity, novelty, duplicate rate, and filter reasons.

## Random-Weight Uni-Mol Control

PR #4 attempted to compare Uni-Mol embeddings from randomly initialized weights, but it also included generated `.pt`, log, cache, and image artifacts. The clean path is the opt-in `random_weight` prediction mode plus an example script:

```bash
python examples/extract_unimol_features.py \
  --model-dir train/test_model \
  --input-csv train/10_mol.csv \
  --output-path runs/random_weight_features/10_mol_features.pt \
  --random-weight \
  --seed 42
```

Keep `runs/`, checkpoints, logs, and generated feature tensors out of Git.
