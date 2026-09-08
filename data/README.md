# Experimental dataset

[experimental_records.csv](experimental_records.csv) contains the unsplit collection of **343 experimental records**, representing **303 distinct raw SMILES strings**. Multiple experiments on the same molecule are retained. Distinct SMILES strings are not a chemistry-normalized molecule count.

The table is reconstructed from the union of `train_pool.csv` and `test_raw.csv` in each of the five [`baselines/datasets/split_seed_0`–`split_seed_4`](../baselines/datasets) archives. All five unions contain exactly the same multiset of observation rows, including original string values and repeated-record multiplicity. Rows are sorted lexically for stable export; this does not recover the original collection order. No split labels are assigned in this table.

## Columns

| Column | Source column | Meaning |
|---|---|---|
| `cas_number` | CAS number | Reported CAS identifier |
| `pubchem_cid` | Pubchem CID | Reported PubChem identifier |
| `name_full` | Name(full) | Full name, when supplied |
| `name_short` | Name(simple) | Short name, when supplied |
| `doi` | DOI number | Source publication DOI |
| `smiles` | SMILES | Molecular structure as stored |
| `initial_pce` | Initial PCE | Reported initial PCE (%) |
| `delta_pce` | ΔPCE | Stored absolute PCE improvement (percentage points) |
| `final_pce` | Final PCE | Reported final PCE (%) |
| `date` | Date | Date as stored in the source |
| `batch` | Batch | Source batch label, retained verbatim |
| `note` | Note | Source note, when supplied |

Blank values remain blank. Numeric precision is preserved as stored. This observation table excludes computed descriptors, duplicated PCE columns, and the seed-dependent `Delta_pred` and `TARGET` fields. Descriptor columns and archived split-specific targets remain available in the baseline split files.

## Export and verification

From the repository root, using Python 3.11 or later:

```bash
python reproducibility/export_unsplit.py
python reproducibility/export_unsplit.py --check
```

The exporter compares observation multisets across all five archives before writing. It preserves experimental rows without deduplicating by SMILES, imputing missing values, or fitting a target transformation.
