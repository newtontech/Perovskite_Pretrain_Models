import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 已有路径
test_path = 'data/split_seed_0/test.csv'
train_path = 'data/split_seed_0/train.csv'

# 读取数据
df_test = pd.read_csv(test_path)
df_train = pd.read_csv(train_path)

# 合并为 all_df
all_df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

# 指定特征列表
feature_list = [
    'C', 'H', 'N', 'F', 'O', 'MW', 'LogP', 'TPSA', 'H_acceptor', 'H_donor',
    'RB', 'Aromatic_rings', 'Aliphatic_rings', 'Saturated_rings', 'Heteroatoms',
    'QED', 'IPC', 'HOMO', 'LUMO', 'Gap', 'Min_ESP', 'Max_ESP', 'Dipole'
]

# 确保只保留存在的特征（防止 KeyError）
features_exist = [f for f in feature_list if f in all_df.columns]

# 提取这些特征的子集
corr_data = all_df[features_exist]

# 计算相关矩阵（Pearson）
corr_matrix = corr_data.corr(method='pearson')

# === 绘制热力图 ===
plt.figure(figsize=(16, 12))  # 可根据特征数量调整大小
sns.heatmap(
    corr_matrix,
    annot=False,           # 是否显示相关系数数值（True 太密，可设 False）
    cmap='coolwarm',       # 颜色方案：红表示正相关，蓝表示负相关
    center=0,              # 中心为 0，对称着色
    square=True,           # 使格子为正方形
    fmt='.2f',             # 数值格式
    cbar_kws={"shrink": 0.8},  # 色条大小
    linewidths=0.5         # 格子间线条
)

plt.title('Feature Correlation Heatmap (Train + Test)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # 自动调整布局
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()