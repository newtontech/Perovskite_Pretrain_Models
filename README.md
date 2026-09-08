# Perovskite Molecular Representation Models

Code, archived datasets, and analysis tools for molecular-modulator prediction in perovskite solar cells. The repository contains Uni-Mol and MolCLR training code, descriptor and fingerprint baselines, and visualization tools.

**Reproducibility status:** the data audit and historical prediction-metric recomputation run without ML dependencies. Full model training and correspondence to all manuscript results have not yet been verified. Known validation and provenance limitations are documented in [Data and reproducibility](reproducibility/README.md).

## Inspect the data and results

With Python 3.11 or later, from the repository root:

```bash
python reproducibility/audit.py --output results/reproducibility
```

This exports complete rows from the selected archived datasets, stored MolCLR fold membership, source checksums, overlap summaries, and metrics recomputed from historical baseline predictions. It does not retrain models or infer missing experimental settings.

- [Reader workbook](reproducibility/ML_reproducibility.xlsx): data records, split and fold summaries, and historical metrics.
- [CSV/JSON snapshot](reproducibility/generated): machine-readable tables generated from checked-in inputs.
- [Data dictionary and limitations](reproducibility/README.md): sample identity, targets, seeds, and interpretation.
- [Code Ocean access and preparation](docs/code_ocean.md): partner eligibility, manuscript association, and private peer review.

Check that the committed tables still match the source content:

```bash
python reproducibility/audit.py --check
```

## Repository layout

| Directory | Purpose |
|---|---|
| `training/` | Uni-Mol training entry points and the bundled `unimol_tools` package |
| `training/molclr/` | MolCLR training, pretrained checkpoints, archived data and folds |
| `baselines/datasets/` | Five archived outer data splits with targets and features |
| `baselines/krfp/` | KRFP feature generation and fingerprint baseline analysis |
| `baselines/predictions/` | Historical baseline prediction records |
| `baselines/figures_dft/`, `baselines/figures_krfp/` | Historical baseline figures |
| `feature_generation/` | Molecular descriptors and quantum-chemistry feature scripts |
| `hyperparameter_search/` | Uni-Mol fine-tuning and hyperparameter search code |
| `visualization/` | UMAP, attention, correlation and molecular visualization |
| `reproducibility/` | Data inventory, reader workbook and audit implementation |
| `perovskite_pretrain/`, `examples/`, `configs/` | Additional experimental workflows and examples |
| `tests/` | Lightweight static and data-integrity checks |

Core split files use `train_pool.csv` for the training/validation pool, `test_raw.csv` for the original test partition, and `test.csv` for its cleaned evaluation subset. The corresponding KRFP arrays use `train_pool_krfp.npy`, `test_raw_krfp.npy`, and `test_krfp.npy`. Source-row identities and stored split contents are explicit in the inventory. Previous names are available in Git history; there are no compatibility directory copies.

## Training code

Full training requires the appropriate ML dependencies, model weights and hardware. `requirements.txt` is an unpinned dependency list, not a verified environment lock. Some archived entry points contain machine-specific paths or device selections that must be configured before running. Do not treat them as a ready-to-run demo.

After resolving the requirements and limitations documented in the reproducibility guide, the historical entry points are:

```bash
# Uni-Mol, from the repository root
cd training
python run.py

# MolCLR, from the repository root
cd training/molclr
python finetune.py
```

These commands identify entry points; they are not evidence that end-to-end training currently passes. Pretrained checkpoints are not interchangeable with final fine-tuned checkpoints for a particular split.

Additional workflows are described in [Pretraining workflows](docs/pretraining_workflows.md). Their lightweight examples are separate from the archived manuscript benchmark.

## Verification

```bash
python -m pip install pytest
python -m pytest -q
python reproducibility/audit.py --check
```

CI runs lightweight checks without downloading training data or model weights. The audit reports historical raw-SMILES overlap within stored MolCLR folds rather than changing those folds to remove the warning. Passing these checks does not establish full training reproducibility.

## Code and data attribution

The code is distributed under the [MIT license](LICENSE). Cite source publications and upstream model frameworks when using their data or representations. A code license alone does not establish redistribution rights for every third-party dataset or checkpoint.

Report results with their data version, target definition, split membership, seeds, environment and checkpoint provenance. Missing settings or unavailable artifacts remain explicitly identified as missing.
