import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP

plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'legend.frameon': False,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize': 16,
    'figure.figsize': (5, 4.5)
})

path = '../train/trainset_points_features_origin_model.pt'
data_dir = '../train/train.csv'

df = pd.read_csv(data_dir)
targets = df['TARGET'].tolist()
features = np.array(torch.load(path).cpu())

umap = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    n_epochs=200
)
embeddings = umap.fit_transform(features)

plt.figure()
scatter = plt.scatter(
    embeddings[:, 0], 
    embeddings[:, 1], 
    c=targets,
    cmap='coolwarm',
    s=30,
    edgecolors='none',
)

cbar = plt.colorbar(scatter)
cbar.set_label('Target value', rotation=270, labelpad=15)
plt.legend(loc='upper right')

plt.title('Pretrained UMAP visualization')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')

plt.savefig('umap.png', dpi=300, bbox_inches='tight')
plt.show()