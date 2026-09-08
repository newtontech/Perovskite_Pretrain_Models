import os
os.environ['UNIMOL_WEIGHT_DIR'] = '/opt/weights' 
from unimol_tools import MolTrain, MolPredict
import joblib
import pandas as pd
import optuna
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging 
import gc
def clear_memory():
    """Clear memory and the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
train_csv_path = 'train_pool.csv'
test_csv_path = 'test_raw.csv'
test_cleaned_csv_path = 'test.csv'
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def format_value(value, max_length=4):
    if isinstance(value, float):
        return f"{value:.4f}"  # Format floating-point values
    elif isinstance(value, str) and len(value) > max_length:
        return value[:max_length]  # Truncate strings
    return str(value)
def tune_hyper_parms_objective(trial, **params) -> float:
    default_params = {
        'batch_size': 32,
        'epochs': 350,
        'kfold': 5,
        'learning_rate': 8.5e-05,
        'load_model_dir': None,
        'logger_level': 1,
        'max_epochs': 100,
        'max_norm': 12.0,
        'metrics': 'mse',
        'model_name': 'unimolv1',
        'model_size': '84m',
        'patience': 60,
        'remove_hs': False,
        'warmup_ratio': 0.03,
        'early_stopping': 60
        }
    all_params = default_params.copy()
    all_params.update(params)

    # Build the output path
    save_path = os.path.join('./hyper_tune', "_".join([f"{format_value(key,4)}_{format_value(value, 4)}" for key, value in all_params.items()]))

    # Check for and load an existing metric.result
    metric_path = os.path.join(save_path, 'metric.result')
    if os.path.exists(metric_path):
        metric = joblib.load(metric_path)
        if 'r2' in metric:
            trial.set_user_attr('r2', metric['r2'])
        return metric['mse']

    # Initialize MolTrain and fit the data
    clf = MolTrain(save_path=save_path, task='regression', data_type='molecule', **all_params)
    clf.fit(train_csv_path)
    mse = clf.model.cv['metric']['mse']
    r2 = clf.model.cv['metric']['r2']

    # Share prediction and saving logic in a helper
    def predict_and_save(clf: MolPredict, csv_path: str, file_name: str):
        result_path = os.path.join(save_path, file_name)
        if not os.path.exists(result_path):
            prediction = clf.predict(csv_path)
            joblib.dump(prediction, result_path)
            return prediction.flatten()
        return joblib.load(result_path).flatten()
    del clf
    clear_memory()
    try:
        clf = MolPredict(load_model=save_path)
        
        # Read all datasets once
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        test_cleaned_df = pd.read_csv(test_cleaned_csv_path)

        # Get predictions
        predicted_test_delta_PCE = predict_and_save(clf, test_csv_path, 'predicted_test_delta_PCE.data')
        predicted_train_delta_PCE = predict_and_save(clf, train_csv_path, 'predicted_train_delta_PCE.data')
        predicted_test_cleaned_delta_PCE = predict_and_save(clf, test_cleaned_csv_path, 'predicted_test_cleaned_delta_PCE.data')

        # Set base metrics
        trial.set_user_attr('r2', r2)
        trial.set_user_attr('mse', mse)
        trial.set_user_attr('train_r2', r2_score(train_df['TARGET'], predicted_train_delta_PCE))
        trial.set_user_attr('train_rmse', root_mean_squared_error(train_df['TARGET'], predicted_train_delta_PCE))
        trial.set_user_attr('test_r2', r2_score(test_df['TARGET'], predicted_test_delta_PCE))
        trial.set_user_attr('test_rmse', root_mean_squared_error(test_df['TARGET'], predicted_test_delta_PCE))
        trial.set_user_attr('test_cleaned_r2', r2_score(test_cleaned_df['TARGET'], predicted_test_cleaned_delta_PCE))
        trial.set_user_attr('test_cleaned_rmse', root_mean_squared_error(test_cleaned_df['TARGET'], predicted_test_cleaned_delta_PCE))

        ib_predicted_train = predicted_train_delta_PCE + train_df['Delta_pred']
        trial.set_user_attr('ib_train_r2', r2_score(train_df['ΔPCE'], ib_predicted_train))
        trial.set_user_attr('ib_train_rmse', root_mean_squared_error(train_df['ΔPCE'], ib_predicted_train))

        ib_predicted_test = predicted_test_delta_PCE + test_df['Delta_pred']
        trial.set_user_attr('ib_test_r2', r2_score(test_df['ΔPCE'], ib_predicted_test))
        trial.set_user_attr('ib_test_rmse', root_mean_squared_error(test_df['ΔPCE'], ib_predicted_test))

        ib_predicted_test_cleaned = predicted_test_cleaned_delta_PCE + test_cleaned_df['Delta_pred']
        trial.set_user_attr('ib_test_cleaned_r2', r2_score(test_cleaned_df['ΔPCE'], ib_predicted_test_cleaned))
        trial.set_user_attr('ib_test_cleaned_rmse', root_mean_squared_error(test_cleaned_df['ΔPCE'], ib_predicted_test_cleaned))
        clear_memory()
    except Exception as e:
        print('Error: ', e)
        # import// logging
        # t=train_df["TARGET"]
        # logging.log(0,msg=f'{t=},{predicted_train_delta_PCE=}')
    return mse

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-6, 2e-4)
    learning_rate = float(f'{learning_rate:.1e}')  
    params = {
        'learning_rate': learning_rate,  # Use a log-uniform distribution
    # Format with one significant digit
        'learning_rate': float(f'{learning_rate:.1e}'),
        'batch_size': trial.suggest_categorical('batch_size', [2,4,8,16,32,64]),
        'epochs': trial.suggest_int('epochs', 50, 400, step=10),
        'early_stopping': trial.suggest_int('early_stopping', 20, 100, step=10),
        'kfold': trial.suggest_categorical('kfold', [5, 10]),
        'max_norm': trial.suggest_discrete_uniform('max_norm', 0,15,1), # Gradient clipping range
        # 'model_size': trial.suggest_categorical('model_size', ['84m', '168m', '336m']),
        # 'metrics': trial.suggest_categorical('metrics', ['mse', 'pearsonr', 'spearmanr', 'mae', 'r2']),
        'warmup_ratio': trial.suggest_discrete_uniform('warmup_ratio', 0,0.12,0.04),  # Use suggest_discrete_uniform
        'remove_hs':False,# trial.suggest_categorical('remove_hs', [True, False]),  # Whether to remove hydrogen atoms
        'split': 'random',#trial.suggest_categorical('split', ['random', 'scaffold']),
        'target_normalize': 'auto',#trial.suggest_categorical('target_normalize', ['auto', 'none', 'minmax', 'standard', 'robust']),
    }
    try:
        return tune_hyper_parms_objective(trial=trial,**params)
    except Exception as e:
        print(e)
        raise optuna.TrialPruned()  # Prune the trial if an error occurs

storage = os.environ['OPTUNA_STORAGE_URL']

# Create an Optuna study
study = optuna.create_study(
    storage=storage,
    study_name="add2_resplit/split_seed_4",
    direction="minimize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=5)
