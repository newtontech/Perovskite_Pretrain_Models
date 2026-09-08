import pandas as pd
import numpy as np

# Dataset paths
test_path = 'datasets/split_seed_0/test_raw.csv'
train_path = 'datasets/split_seed_0/train_pool.csv'

# Read data
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

# Combine into all_df
all_df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

# === Check that TARGET exists ===
if 'TARGET' not in all_df.columns:
    raise ValueError("The data must contain a 'TARGET' column")

# Define the molecular feature list
feature_list = [
    'C', 'H', 'N', 'F', 'O', 'MW', 'LogP', 'TPSA', 'H_acceptor', 'H_donor',
    'RB', 'Aromatic_rings', 'Aliphatic_rings', 'Saturated_rings', 'Heteroatoms',
    'QED', 'IPC', 'HOMO', 'LUMO', 'Gap', 'Min_ESP', 'Max_ESP', 'Dipole'
]

# Require available features and nonmissing TARGET
features_exist = [f for f in feature_list if f in all_df.columns]
data_with_target = all_df[features_exist + ['TARGET']].dropna()

X = data_with_target[features_exist]
y = data_with_target['TARGET']

# === Stage 1: Pearson correlation of each feature with TARGET ===
corr_with_target = X.corrwith(y, method='pearson')
corr_with_target_abs = corr_with_target.abs().sort_values(ascending=False)

print("Feature correlations with TARGET (descending):")
print(corr_with_target_abs.head(20))

# Keep the top N features by correlation with TARGET
top_n = 20
selected_from_corr = corr_with_target_abs.head(top_n).index.tolist()
X_top = X[selected_from_corr].copy()

print(f"\nStage 1: keep the features most correlated with TARGET: {len(selected_from_corr)} features")

# === Stage 2: Remove correlated features without clustering ===
# Compute absolute pairwise feature correlations
corr_matrix = X_top.corr(method='pearson').abs()

# Collect features to remove
to_drop = set()

# Inspect the upper triangle to avoid duplicate pairs
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        feat_i = corr_matrix.columns[i]
        feat_j = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]

        if corr_value > 0.75:  # High-correlation threshold
            # Compare absolute correlations of the two features with TARGET
            corr_i_target = abs(corr_with_target[feat_i])
            corr_j_target = abs(corr_with_target[feat_j])

            # Remove the feature with weaker correlation to TARGET
            to_remove = feat_i if corr_i_target < corr_j_target else feat_j
            to_drop.add(to_remove)

# Remove the selected features
final_features = [f for f in X_top.columns if f not in to_drop]
final_features_sorted = sorted(final_features, key=lambda x: abs(corr_with_target[x]), reverse=True)

print(f"\nAmong the top {top_n} features, {len(to_drop)} features were removed for high correlation:")
print(sorted(to_drop))

print(f"\nSelected {len(final_features_sorted)} features after TARGET correlation ranking and redundancy removal:")
print(final_features_sorted)