# AI for Perovskite Solar Cells


## Quick Start

### Easy Installation

```bash
# clone the repository
git clone https://github.com/newtontech/Perovskite_Pretrain_Models.git

# create virtual env
conda create -n aifp python==3.11
conda activate aifp
pip install -r requirements.txt
```

### Training

#### Uni-Mol Training

You can run a quick training script from pretrained Uni-Mol model by:
```bash
cd train
python run.py
```

### Visualization

#### Features Visualization

First, you can save the features of a given set of molecules by:
```bash
cd train
python get_features.py
```

Especially, if you want to save the features of a given set of molecules by a model with random initialized weights, do:

```bash
python get_random_features.py
```

After saving the point features to a .pt file, you can run by:

```bash
cd visualize
python draw_umap.py
```

If you have two sets of features and want to highlight one of them, run by:
```bash
cd visualize
python draw_umap_with_additional_points.py
```

#### Heatmap Visualization

First, you can save the heatmap and atom list of a given molecule by:
```bash
cd train
python get_heatmap.py
```

After that, run:
```bash
cd visualize
python draw_heatmap.py
```

### Baseline Methods

#### MolClR

You can run a finetune process of MolCLR model by
```bash
cd train/train_molclr
python finetune.py
```

After that, run the data post-process script by

```bash
python collect_data.py
```

Finally, get the visualization result by following instructions specified in

```bash
draw.ipynb
```



#### DFT-Features

You can run the feature heatmap visualization of DFT by
```bash
cd baselines
python draw_correlation.py
```

, and run the feature selection process by
```bash
python feature_selection_cluster.py
```

Run the baseline models of DFT with a random search of hyperparams by
```bash
python baseline_search_get.py
```

After that,all the prediction results are saved at \predictions

You can get the visualization result of the best performance model by running
```bash
python draw_best_results.py
```
And the corresponding images are saved in /scatter_img

#### KRFP Features

Run the data generation pipeline by
```bash
cd data_krfp
python generate_krfp.py
```
And the data is saved to /data folder

After that, run the baseline models of KRFP-features with a random search of hyperparams by
```bash
python baseline_search_get.py
```

After that,all the prediction results are saved at \predictions_krfp
You can get the visualization result of the best performance model by running
```bash
python draw_best_results.py
```
And the corresponding images are saved in /scatter_img_krfp