# 🌞 AI for Perovskite Solar Cells

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Accelerating Perovskite Solar Cell Discovery with Pre-trained Molecular Representation Learning**

This repository implements state-of-the-art deep learning models for predicting perovskite solar cell properties, combining **pre-trained molecular representations** with **domain-specific fine-tuning**.

---

## ✨ Highlights

- **🤖 Multiple Pre-trained Models**: Uni-Mol, MolCLR, and more for molecular representation learning
- **📊 Comprehensive Baselines**: DFT features, KRFP fingerprints with ML baselines (XGBoost, Random Forest, SVR)
- **🎨 Rich Visualization Tools**: UMAP embeddings, attention heatmaps, and correlation analysis
- **🔬 Domain-Specific**: Tailored for perovskite material property prediction

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/newtontech/Perovskite_Pretrain_Models.git
cd Perovskite_Pretrain_Models

# Create conda environment
conda create -n aifp python=3.11 -y
conda activate aifp

# Install dependencies
pip install -r requirements.txt
```

### Training Your First Model

```bash
cd train
python run.py
```

## Reproducibility Checklist

Use this as the public path for reproducing results from the repository:

1. Create the Python 3.11 environment and install `requirements.txt`.
2. Record the dataset source, target property, train/validation/test split, random seed, and metric definition before each run.
3. Run baseline feature generation before comparing pretrained models against DFT or KRFP baselines.
4. Save generated metrics and figures beside the run output, then update the README result table only from reviewed artifacts.
5. Keep private datasets and large pretrained weights outside Git; publish checksums and download instructions when an artifact can be shared.

### Minimal Demo Path

```bash
conda create -n perovskite-pretrain python=3.11 -y
conda activate perovskite-pretrain
pip install -r requirements.txt

cd train
python run.py
```

If a full dataset or pretrained checkpoint is unavailable locally, treat this as a smoke test and document the missing artifact in the experiment notes.

### CI Smoke Checks

GitHub Actions runs a lightweight smoke suite that does not install ML dependencies, download datasets, or fetch model weights:

```bash
python -m pip install pytest
pytest -q
```

The checks compile Python sources without importing heavy modules and prevent new tracked generated artifacts under `__pycache__/`, `logs/`, or `checkpoints/`.

### Data and Model Artifacts

- Public datasets: cite the paper, database, DOI, or upstream repository in the experiment notes.
- Private datasets: store outside this repository and keep only schema, feature definitions, and anonymized examples here.
- Pretrained weights: store in release assets, object storage, or an institutional data repository before linking from the README.
- Generated outputs: keep final plots and metrics under a dated run folder; copy only publication-ready figures into `visualize/` or docs.

### Pretraining Workflow Stubs

Open pretraining requests are tracked as lightweight, reproducible workflow stubs rather than committed model artifacts:

- ChemBERTa2 training details: `docs/pretraining_workflows.md` and `configs/pretraining/chemberta2_mlm.json`
- Property prediction: `configs/pretraining/property_prediction_multitask.json`
- Molecular generation: `configs/pretraining/molecule_generation_vae.json`
- Random-weight Uni-Mol control: `examples/extract_unimol_features.py`

---

## 📁 Project Structure

```
Perovskite_Pretrain_Models/
├── train/                          # Main training scripts
│   ├── run.py                      # Uni-Mol training entry point
│   ├── get_features.py             # Feature extraction
│   ├── get_heatmap.py              # Attention heatmap generation
│   └── train_molclr/               # MolCLR fine-tuning
├── baselines/                      # Baseline methods
│   ├── baseline_search_get.py      # Hyperparameter search
│   ├── feature_selection_cluster.py
│   └── data_krfp/                  # KRFP feature generation
├── visualize/                      # Visualization tools
│   ├── draw_umap.py                # UMAP visualization
│   ├── draw_heatmap.py             # Heatmap rendering
│   └── draw_correlation.py         # Feature correlation
└── rdkit_dft_features_generation/  # DFT feature extraction
```

---

## 🎯 Features

### Model Training

#### Uni-Mol Fine-tuning
State-of-the-art 3D molecular representation learning pretrained on large-scale molecular datasets.

```bash
cd train
python run.py
```

#### MolCLR Fine-tuning
Contrastive learning framework for molecular representations.

```bash
cd train/train_molclr
python finetune.py
python collect_data.py
# View results in draw.ipynb
```

### Visualization

#### UMAP Embedding Visualization
Visualize high-dimensional molecular features in 2D space.

```bash
cd train
python get_features.py          # Save features
cd ../visualize
python draw_umap.py             # Basic UMAP
python draw_umap_with_additional_points.py  # Highlight specific molecules
```

#### Attention Heatmaps
Understand which atoms the model focuses on for predictions.

```bash
cd train
python get_heatmap.py           # Generate heatmap data
cd ../visualize
python draw_heatmap.py          # Render visualization
```

### Baseline Methods

#### DFT + ML Models
Traditional machine learning with Density Functional Theory features.

```bash
cd baselines

# Feature correlation analysis
python draw_correlation.py

# Feature selection
python feature_selection_cluster.py

# Train & evaluate baselines
python baseline_search_get.py

# Visualize best results
python draw_best_results.py
```

#### KRFP Fingerprints
Kernel-based molecular fingerprints with ML baselines.

```bash
cd baselines/data_krfp
python generate_krfp.py         # Generate KRFP features

cd ..
python baseline_search_get.py   # Train models
python draw_best_results.py     # Visualize
```

---

## 📊 Results

| Model | Features | Test RMSE | Validation R² |
|-------|----------|-----------|---------------|
| Uni-Mol | 3D Structure | - | - |
| MolCLR | Graph | - | - |
| XGBoost | DFT | - | - |
| Random Forest | KRFP | - | - |

Update this table only from a completed run that includes environment, data source, split, seed, and output artifacts.

## Citation

If this repository helps your research, cite the associated perovskite pretraining work and the upstream model frameworks used in your experiment.

```bibtex
@software{perovskite_pretrain_models,
  title = {AI for Perovskite Solar Cells: Pretrained Molecular Representation Models},
  author = {Yan, Haoming and contributors},
  year = {2026},
  url = {https://github.com/newtontech/Perovskite_Pretrain_Models}
}
```

---

## 📝 Requirements

- Python 3.11+
- PyTorch 2.0+
- RDKit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- umap-learn

See `requirements.txt` for complete dependencies.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Uni-Mol team for the pretrained model
- MolCLR framework authors
- The perovskite research community

---

<div align="center">

**Made with ❤️ for Accelerating Materials Discovery**

[⭐ Star this repo](https://github.com/newtontech/Perovskite_Pretrain_Models) • [🐛 Report Issues](https://github.com/newtontech/Perovskite_Pretrain_Models/issues) • [💡 Feature Requests](https://github.com/newtontech/Perovskite_Pretrain_Models/issues)

</div>
