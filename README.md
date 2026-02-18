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
