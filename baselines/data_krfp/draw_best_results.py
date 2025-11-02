import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.cm as cm

# 自定义上标（用于 Matplotlib 数学文本）
from matplotlib.mathtext import _mathtext as mathtext
mathtext.FontConstantsBase.sup1 = 0.5

# === 全局样式设置（已生效）===
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
    'figure.figsize': (10, 9),  # ← 使用这个尺寸
    'savefig.dpi': 300,           # ← 保存图像的默认 DPI
    'figure.dpi': 100,            # ← 显示用 DPI
    'axes.linewidth': 1.2,        # 坐标轴线宽
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'lines.markersize': 6         # 散点大小统一
})

# 自定义颜色
blue = cm.Blues(0.8)
red = cm.Reds(0.8)
gray = 'gray'

alpha = 0.3  # 用于填充或次要元素（可选）
# ==============================


def plot_train_predictions_best_seed(model_name, prediction_dir='predictions_krfp', sets=['train', 'val', 'test_cleaned']):
    """
    对指定模型（如 SVR, Random Forest），在所有 seed 中选择 test R² 最高的 seed，
    然后绘制其指定 split 的预测 vs 真实值散点图 + 回归线，并标注性能指标。

    参数:
        model_name (str): 模型名称，如 'SVR', 'Random Forest', 'XGBoost'
        prediction_dir (str): 存放预测 CSV 文件的目录
        sets (list): 要绘制的 split 列表，如 ['train', 'val', 'test']
    """
    pattern = os.path.join(prediction_dir, f'{model_name}_seed_*.csv')
    filepaths = sorted(glob.glob(pattern))
    if not filepaths:
        raise FileNotFoundError(f"❌ No files found matching pattern: {pattern}")

    print(f"🔍 Found {len(filepaths)} files for model '{model_name}'")

    test_scores = []
    for filepath in filepaths:
        try:
            seed = int(filepath.split('seed_')[-1].replace('.csv', ''))
        except:
            print(f"⚠️  Could not parse seed from {filepath}, skipping.")
            continue

        df = pd.read_csv(filepath)
        df_test = df[df['split'] == 'test_cleaned']
        if len(df_test) == 0:
            print(f"⚠️  No test data in {filepath}")
            r2 = -np.inf
            rmse = np.inf
        else:
            y_true = df_test['true']
            y_pred = df_test['pred']
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        test_scores.append({'seed': seed, 'filepath': filepath, 'r2': r2, 'rmse': rmse})

    scores_df = pd.DataFrame(test_scores)
    best_row = scores_df.loc[scores_df['r2'].idxmax()]
    best_seed = best_row['seed']
    best_filepath = best_row['filepath']
    best_test_r2 = best_row['r2']
    best_test_rmse = best_row['rmse']

    print(f"✅ Best test R² for '{model_name}' is from seed {best_seed}: R² = {best_test_r2:.4f}, RMSE = {best_test_rmse:.4f}")

    # 读取最佳文件
    df_best = pd.read_csv(best_filepath)

    for setname in sets:
        df_set = df_best[df_best['split'] == setname]
        if len(df_set) == 0:
            print(f"⚠️  No {setname} data in {best_filepath}")
            continue

        x = df_set['true']
        y = df_set['pred']

        r2 = r2_score(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))

        # === 开始绘图（使用 rcParams 设置）===
        fig, ax = plt.subplots()  # 自动使用 figsize=(5,4.5)

        # 散点图（使用自定义蓝色）
        ax.scatter(x, y, alpha=0.8, color=blue, edgecolor='none', s=50, label=f'{setname.capitalize()}')

        # 拟合回归线（使用自定义红色）
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color=red, linewidth=2.5, label='Fit: $\\mathdefault{y=ax+b}$')

        # 理想线 y = x（灰色）
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], '--', color=gray, linewidth=2, label='Ideal: $\\mathdefault{y=x}$')

        # 标注性能指标（使用 mathtext 支持 bold/italic）
        text_str = f'$\\mathbf{{{setname.capitalize()}\\ R^2 = {r2:.4f}}}$\n$\\mathbf{{{setname.capitalize()}\\ RMSE = {rmse:.4f}}}$'.replace("_", "\,")
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', bbox=props, ha='left')

        # 坐标轴标签
        ax.set_xlabel('True Values', fontsize=18)
        ax.set_ylabel('Predicted Values', fontsize=18)
        ax.set_title(f'{model_name}\n{setname.capitalize()} Set', fontsize=18, pad=10)

        # 图例
        ax.legend(fontsize=14, loc='lower right')

        # 网格（可选，轻量级）
        ax.grid(True, alpha=0.2)

        # 布局优化
        plt.tight_layout()

        # 保存图像（使用 300 dpi）
        base_dir = '../scatter_img_krfp'
        output_fig = f'{base_dir}/{model_name}_{setname}_scatter_best_seed.png'
        plt.savefig(output_fig, dpi=None, bbox_inches='tight')  # dpi 已在 rcParams 中设置
        print(f"📈 {setname.capitalize()} scatter plot saved as '{output_fig}'")

        plt.show()


# === 使用示例 ===
plot_train_predictions_best_seed('SVR')
plot_train_predictions_best_seed('Random Forest')
plot_train_predictions_best_seed('XGBoost')
plot_train_predictions_best_seed('ElasticNet')