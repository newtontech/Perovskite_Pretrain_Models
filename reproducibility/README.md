# Data and reproducibility

The [unsplit experimental dataset](../data/experimental_records.csv) contains all 343 observation records before assigning training or test roles. See its [data dictionary and reconstruction method](../data/README.md).

This directory makes the archived data splits and prediction records inspectable. It does not establish that every archived result corresponds to the final manuscript, and it does not provide a verified full training reproduction.

## Start here

From the repository root, with Python 3.11 or later:

```bash
python reproducibility/audit.py --output results/reproducibility
```

The audit uses only the Python standard library. It exports source records, stored split membership, file checksums, and summaries, and recomputes metrics from saved baseline predictions. It does not train models, generate new splits, or run inference from checkpoints. Generated outputs belong in `results/reproducibility`.

| Output | Contents |
|---|---|
| `samples.csv` | Every baseline dataset row and MolCLR training-pool row, with file/row identity, seed, role, and every source column prefixed with `source:` |
| `file_manifest.csv` | Source paths, record counts, byte hashes and normalized-content hashes |
| `split_summary.csv` | Per-seed record and raw-SMILES counts, training overlap and cleaned-test checks |
| `fold_membership.csv` | Stored MolCLR train/validation members by seed and fold, with source-row mappings |
| `fold_summary.csv` | Stored fold sizes and raw-SMILES overlap |
| `historical_metrics.csv` | R², RMSE and MAE calculated from saved baseline predictions, retaining their split labels |
| `summary.json` | Machine-readable audit summary |

`record_id` uses the source path and row number; `source_index_zero_based` records the positional index used in stored fold files. `raw_smiles_id` describes string identity only. The export role `test_cleaned` refers to the current `test.csv`, and `molclr_train_pool` distinguishes the MolCLR source from the baseline source. Metric rows retain historical labels, including `val`; read their limitations below. Undefined R² is blank. Normalized hashes support content comparisons across line-ending changes, while byte hashes identify the exact file bytes.

To compare regenerated tables against an existing output directory without rewriting it:

```bash
python reproducibility/audit.py --output results/reproducibility --check
```

This comparison enforces normalized source content and generated tables, allowing byte-hash differences caused by line endings. The companion `ML_reproducibility.xlsx` presents the same audit tables for browsing; CSV/JSON remain the machine-readable outputs.

| Directory | Purpose |
|---|---|
| `baselines/datasets` | Five archived outer splits, including source fields, targets and descriptors |
| `baselines/krfp` | Fingerprint baseline materials |
| `training/molclr` | MolCLR code, stored fold indices and historical results |
| `feature_generation` | Descriptor and feature generation code |
| `hyperparameter_search` | Uni-Mol fine-tuning and search materials |
| `visualization` | Figure and representation analysis code |
| `reproducibility` | Data audit, reader tables and reproducibility documentation |

## Archived outer splits

The following are record counts, not counts of unique molecules. `train_pool.csv` is the pool used for inner training and validation; `test_raw.csv` is the original test partition, and `test.csv` is its cleaned evaluation subset. Each seed describes a repeated outer split; these are not five mutually exclusive test folds. Renaming these files describes their roles and does not change the underlying experimental algorithm.

| Outer seed | Training/validation pool | Original test | Cleaned test |
|---:|---:|---:|---:|
| 0 | 291 | 52 | 39 |
| 1 | 291 | 52 | 47 |
| 2 | 291 | 52 | 40 |
| 3 | 291 | 52 | 40 |
| 4 | 291 | 52 | 43 |

In these files, the cleaned test is obtained by removing test records whose **raw SMILES strings** occur in the training pool. The audit checks this relationship against the stored rows. This establishes string-level separation only: chemically equivalent SMILES, salts, stereochemistry and molecule groups still require a chemistry-aware identity audit. Identical SMILES can also represent different experimental records, so SMILES alone is not a unique sample identifier.

The archived test-side removal must be reconciled with the author-approved experimental protocol before these files are described as the final manuscript splits. Historical tables from other dataset versions must not be substituted merely because their names or split ratios look similar.

All available source columns are retained in the record export, including identifiers, DOI, PCE fields and descriptors. Blank source values remain blank. Exported record identifiers describe file/row provenance; they do not invent a globally matched experimental identity.

## Targets, folds and random seeds

The baseline `TARGET` is a residual target: the stored values follow `Final_PCE - Initial_PCE - Delta_pred`. It must not be interpreted as unadjusted PCE improvement. The origin and fitting population of `Delta_pred` still need to be bound to the final experiment. Baseline metric recomputation uses the archived `true` and `pred` values in this residual space; it is not new model evaluation or a reconstruction of every manuscript figure.

MolCLR has stored ten-fold index files for the outer seeds. Exporting these indices documents their contents, not proof that a historical run used them. The [loader](../training/molclr/dataset/dataset_test.py) reads a hard-coded external JSON location, while the [fine-tuning entry point](../training/molclr/finetune.py) defaults to outer seed 0. These runtime choices must be reconciled with the per-seed archives and the intended fold count.

The [baseline search](../baselines/baseline_search_get.py) uses five-fold shuffled cross-validation with seed 42 and a search seed of 42. These are distinct from outer seed labels 0–4. Model initialization, data-loader ordering, conformer generation and any other random processes require their own recorded settings. Unknown historical seeds remain unknown; the audit does not fill them with a default.

## Interpretation limits

- **Baseline validation:** the [search script](../baselines/baseline_search_get.py) selects the first validation fold, then predicts it after refitting the model on the full training pool. The saved `val` rows are therefore not independent held-out or out-of-fold predictions. Scaling is also fitted before cross-validation. Archived outputs are preserved; these issues require a separately validated training correction.
- **MolCLR validation:** [`_validate`](../training/molclr/finetune.py) iterates the concatenated loader datasets rather than their sampled validation members. This includes training records and affects checkpoint selection. Stored validation results are not evidence of a clean validation procedure.
- **Weights and provenance:** an upstream pretrained representation checkpoint is not a final per-seed fine-tuned checkpoint. Complete model/configuration/input/checkpoint/prediction links have not been established for all reported experiments.
- **Figure-source data:** the separately maintained `data_collection` provenance work organizes figure inputs and their processing. It did not archive all machine-learning folds, training configurations and model weights. Historical data versions in that collection cannot silently replace this benchmark.
- **Environment:** the audit needs no ML dependencies, but full model training still requires a verified dependency environment, portable paths and available data/weights. A passing audit or smoke test does not demonstrate successful model training.

The per-split MolCLR tuning scripts obtain their Optuna database connection from `OPTUNA_STORAGE_URL`. Supply it through your local environment or your compute platform's secret settings; never commit connection credentials. Credentials previously embedded in repository history must be rotated by the database owner. This update removes them from the current scripts without rewriting history.

## What is needed for full reproduction

For each final experiment, retain the original source record mapping, target-generation procedure, outer and inner membership, all random settings, preprocessing fitted only within the appropriate training partition, model hyperparameters, code revision, environment and input/checkpoint checksums. Bind every saved prediction and manuscript figure to those records. Preserve historical artifacts separately from corrected analyses and report any resulting numerical differences.

For platform access and capsule preparation, see [Code Ocean](../docs/code_ocean.md).
