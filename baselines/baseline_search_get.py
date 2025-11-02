import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from predict_and_save import save_predictions_with_true
import warnings
warnings.filterwarnings('ignore')

# 特征列表
feature_list = ['LUMO', 'Heteroatoms', 'Gap', 'Max_ESP', 'H_acceptor', 'TPSA', 'MW', 'F', 'Aromatic_rings',
                'N', 'HOMO', 'RB', 'H', 'QED', 'IPC', 'Saturated_rings', 'Dipole']

# 模型定义（移出循环，避免重复定义）
models_params = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'use_scaler': False
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42, max_iter=10000),
        'params': {
            'alpha': np.logspace(-4, 1, 100),
            'l1_ratio': np.linspace(0.1, 0.9, 9)
        },
        'use_scaler': True
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': np.logspace(-2, 3, 100),
            'epsilon': np.logspace(-3, 1, 50),
            'kernel': ['rbf'],
            'gamma': np.logspace(-4, -1, 20)
        },
        'use_scaler': True
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': np.logspace(-3, -1, 100),
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': np.logspace(-3, 1, 100),
            'reg_lambda': np.logspace(-2, 2, 100),
        },
        'use_scaler': False
    }
}

# 存储每个模型在每个 seed 上的 test 性能
all_results = {model: {'test_r2': [], 'test_rmse': []} for model in models_params.keys()}

# 外层循环：5 个不同的数据划分 seed
for seed in range(5):
    print(f"\n{'='*60}")
    print(f"🔁 Processing Seed {seed}")
    print(f"{'='*60}")

    test_path = f'data/split_seed_{seed}/test_cleaned.csv'
    train_path = f'data/split_seed_{seed}/train.csv'

    # 读取数据
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 转为 numpy
    X_train = np.array(df_train[feature_list])
    y_train = np.array(df_train['TARGET'])
    X_test = np.array(df_test[feature_list])
    y_test = np.array(df_test['TARGET'])
    smiles_train = df_train['SMILES'].values
    smiles_test = df_test['SMILES'].values

    # 5折划分（用于调参）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    #get the first fold of validation set as x_val and y_val
    for train_index, val_index in kf.split(X_train):
        X_val, y_val = X_train[val_index], y_train[val_index]
        smiles_val = smiles_train[val_index]
        break

    # 对每个模型进行调参和测试
    for name, config in models_params.items():
        print(f"🔍 Optimizing {name}...")

        model = config['model']
        params = config['params']
        use_scaler = config['use_scaler']

        # 是否标准化
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test

        # RandomizedSearchCV：在5折CV上选择最佳超参数
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=100,
            scoring='r2',
            cv=kf,
            refit=True,
            n_jobs=-1,
            random_state=42,
            verbose=20
        )
        search.fit(X_train_scaled, y_train)

        # 在 test 上预测

        y_train_pred = search.best_estimator_.predict(X_train_scaled)
        y_val_pred = search.best_estimator_.predict(X_val_scaled)       # ✅ 预测 val
        y_test_pred = search.best_estimator_.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # 保存该 seed 的结果
        all_results[name]['test_r2'].append(test_r2)
        all_results[name]['test_rmse'].append(test_rmse)

        print(f"  Seed {seed} - {name} → Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        save_predictions_with_true(
            model_name=name,
            seed=seed,
            smiles_train=smiles_train,
            smiles_val=smiles_val,
            smiles_test=smiles_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_val_pred=y_val_pred,
            y_test_pred=y_test_pred
        )

# 最终汇总：每个模型在 5 seeds 上的平均性能
print("\n" + "="*70)
print("🏆 FINAL RESULTS (5 Seeds Average ± Std)")
print("="*70)
for name in all_results.keys():
    r2_mean = np.mean(all_results[name]['test_r2'])
    r2_std = np.std(all_results[name]['test_r2'], ddof=1)
    rmse_mean = np.mean(all_results[name]['test_rmse'])
    rmse_std = np.std(all_results[name]['test_rmse'], ddof=1)
    print(f"{name}:")
    print(f"  Test R²  = {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"  Test RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")