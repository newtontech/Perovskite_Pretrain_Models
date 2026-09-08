import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset paths
test_path = 'datasets/split_seed_0/test_raw.csv'
train_path = 'datasets/split_seed_0/train_pool.csv'

# Read data
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

# Combine into all_df
all_df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

# Specify the feature list
feature_list = [
    'C', 'H', 'N', 'F', 'O', 'MW', 'LogP', 'TPSA', 'H_acceptor', 'H_donor',
    'RB', 'Aromatic_rings', 'Aliphatic_rings', 'Saturated_rings', 'Heteroatoms',
    'QED', 'IPC', 'HOMO', 'LUMO', 'Gap', 'Min_ESP', 'Max_ESP', 'Dipole'
]

# Keep only available features to prevent KeyError
features_exist = [f for f in feature_list if f in all_df.columns]

# Select the feature subset
corr_data = all_df[features_exist]

# Compute the Pearson correlation matrix
corr_matrix = corr_data.corr(method='pearson')

# === Plot the heatmap ===
plt.figure(figsize=(16, 12))  # Adjust size for the feature count
sns.heatmap(
    corr_matrix,
    annot=False,           # Show coefficient values; disable for dense plots
    cmap='coolwarm',       # Red for positive and blue for negative correlations
    center=0,              # Center the color scale at zero
    square=True,           # Use square cells
    fmt='.2f',             # Number format
    cbar_kws={"shrink": 0.8},  # Colorbar size
    linewidths=0.5         # Cell border width
)

plt.title('Feature Correlation Heatmap (Train + Test)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # Adjust the layout automatically
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()