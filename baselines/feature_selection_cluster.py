import pandas as pd
import numpy as np

# 已有路径
test_path = 'data/split_seed_0/test.csv'
train_path = 'data/split_seed_0/train.csv'

# 读取数据
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

# 合并为 all_df
all_df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

# === 确保 TARGET 列存在 ===
if 'TARGET' not in all_df.columns:
    raise ValueError("请确认数据中包含 'TARGET' 列")

# 假设 feature_list 是你的分子特征
feature_list = [
    'C', 'H', 'N', 'F', 'O', 'MW', 'LogP', 'TPSA', 'H_acceptor', 'H_donor',
    'RB', 'Aromatic_rings', 'Aliphatic_rings', 'Saturated_rings', 'Heteroatoms',
    'QED', 'IPC', 'HOMO', 'LUMO', 'Gap', 'Min_ESP', 'Max_ESP', 'Dipole'
]

# 确保特征存在且无缺失 TARGET
features_exist = [f for f in feature_list if f in all_df.columns]
data_with_target = all_df[features_exist + ['TARGET']].dropna()

X = data_with_target[features_exist]
y = data_with_target['TARGET']

# === 阶段 1：计算每个特征与 TARGET 的相关性（Pearson）===
corr_with_target = X.corrwith(y, method='pearson')
corr_with_target_abs = corr_with_target.abs().sort_values(ascending=False)

print("特征与 TARGET 的相关性（降序）：")
print(corr_with_target_abs.head(20))

# 保留与 TARGET 相关性最高的 top N 个特征
top_n = 20
selected_from_corr = corr_with_target_abs.head(top_n).index.tolist()
X_top = X[selected_from_corr].copy()

print(f"\n阶段1：保留与 TARGET 相关性最高的 {len(selected_from_corr)} 个特征")

# === 阶段2：去除特征间高相关性（不使用聚类）===
# 计算特征间相关性矩阵（绝对值）
corr_matrix = X_top.corr(method='pearson').abs()

# 存储需要删除的特征
to_drop = set()

# 只检查上三角（避免重复对）
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        feat_i = corr_matrix.columns[i]
        feat_j = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]

        if corr_value > 0.75:  # 高相关阈值
            # 比较这两个特征谁与 TARGET 的相关性更强（绝对值）
            corr_i_target = abs(corr_with_target[feat_i])
            corr_j_target = abs(corr_with_target[feat_j])

            # 删除与 TARGET 相关性较弱的那个
            to_remove = feat_i if corr_i_target < corr_j_target else feat_j
            to_drop.add(to_remove)

# 执行删除
final_features = [f for f in X_top.columns if f not in to_drop]
final_features_sorted = sorted(final_features, key=lambda x: abs(corr_with_target[x]), reverse=True)

print(f"\n在 top {top_n} 个特征中，发现 {len(to_drop)} 个因高相关而被删除的特征：")
print(sorted(to_drop))

print(f"\n✅ 最终选出 {len(final_features_sorted)} 个特征（高相关 TARGET + 去除冗余）:")
print(final_features_sorted)