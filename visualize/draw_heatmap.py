import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

root_dir = '../train/single_mol_heatmap'
heatmap_feature_dir = os.path.join(root_dir, 'mean_heatmap.pt')
atom_list_dir = os.path.join(root_dir, 'atoms.json')

heatmap = torch.load(heatmap_feature_dir)
heatmap = heatmap[0][1:-1, 1:-1]

with open(atom_list_dir, 'r') as f:
    atom_list = json.load(f)

heatmap_np = heatmap.detach().cpu().numpy()
heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    heatmap_norm,
    cmap="viridis",
    annot=False,
    square=True,
    xticklabels=atom_list,
    yticklabels=atom_list,
    cbar=True,
)

plt.title("Atomic Interaction Heatmap")
ax.tick_params(
    axis='both',
    which='both',
    length=0,
    bottom=True,
    left=True,
)
plt.xlabel("Atom index")
plt.ylabel("Atom index")

plt.savefig('heatmap.png')