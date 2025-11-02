import pandas as pd
import os

def save_predictions_with_true(model_name, seed,smiles_train,smiles_val,smiles_test, y_train, y_val, y_test,
                               y_train_pred, y_val_pred, y_test_pred,
                               output_dir='predictions'):
    """
    将训练集、验证集、测试集的预测结果与真实值保存为 CSV 文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{model_name}_seed_{seed}.csv')

    # 合并为 DataFrame
    results = []
    for split, y_true, y_pred,smile in zip(
        ['train', 'val', 'test_cleaned'],
        [y_train, y_val, y_test],
        [y_train_pred, y_val_pred, y_test_pred],
        [smiles_train, smiles_val, smiles_test]
    ):
        df_split = pd.DataFrame({
            'SMILES': smile,
            'split': split,
            'true': y_true,
            'pred': y_pred, 
        })
        results.append(df_split)

    # 拼接所有 split
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(filepath, index=False)
    print(f"  🔽 Predictions saved to {filepath}")