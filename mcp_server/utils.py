import os
import csv
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import pandas as pd

from unimol_tools import MolTrain, MolPredict
import logging
logger = logging.getLogger(__name__)
from rdkit import Chem


def _default_params() -> Dict[str, Any]:
    return {
        "amp": True,
        "anomaly_clean": True,
        "batch_size": 1,
        "cuda": True,
        "data_type": "molecule",
        "epochs": 350,
        "freeze_layers": 25,
        "freeze_layers_reversed": 25,
        "kfold": 5,
        "learning_rate": 8.5e-05,
        "load_model_dir": None,
        "logger_level": 1,
        "max_epochs": 100,
        "max_norm": 12.0,
        "metrics": "mse",
        "model_name": "unimolv1",
        "model_size": "84m",
        "num_classes": 1,
        "patience": 60,
        "remove_hs": False,
        "seed": 42,
        "smi_strict": True,
        "smiles_col": "SMILES",
        "split": "random",
        "split_group_col": "scaffold",
        "split_seed": 42,
        "target_col_prefix": "TARGET",
        "target_cols": "TARGET",
        "target_normalize": "auto",
        "task": "regression",
        "use_amp": True,
        "use_cuda": True,
        "warmup_ratio": 0.03,
        "early_stopping": 80,
        "train_from_scratch": False,
        "save_dir": "./exp"
    }


def SMILESListToCSV(smiles_list: List[str]) -> Path:
    """
    Convert a list of SMILES strings to CSV file

    Args:
        smiles_list: List of SMILES strings to convert

    Returns:
        Path to the created CSV file
    """

    # Generate timestamp-based filename in test folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)
    filename = test_dir / f"{timestamp}_smiles.csv"

    # Create final file with proper filename
    final_path = filename

    # Create DataFrame and write to CSV using pandas
    df = pd.DataFrame({
        'SMILES': smiles_list
    })
    df.to_csv(final_path, index=False)

    return final_path

def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    Filter a list of SMILES strings to keep only valid ones.

    Args:
        smiles_list: List of SMILES strings to filter

    Returns:
        List of valid SMILES strings
    """
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            # Optionally log invalid SMILES
            print(f"Warning: Invalid SMILES '{smiles}' removed from list")
    return valid_smiles


def UniMolTrain(csv_path: Path, params: Dict[str, Any]) -> Dict[str, Path]:
    """
    Train a UniMol molecular property model from a CSV and write all artifacts to save_dir.

    Inputs
    - csv_path (Path): Training CSV. Required columns:
        * SMILES column (name via params['smiles_col'], default 'SMILES')
        * Target column(s) (via params['target_cols'] or names with params['target_col_prefix']; default 'TARGET')
      Optional: grouping/split column named by params['split_group_col'] (e.g., 'scaffold').
    - save_dir (str): Output directory for checkpoints/config/logs.
    - params (Dict[str, Any]): Optional overrides; keys and defaults listed below.

    Parameters (defaults shown)
    - amp: True; use_amp: True
    - cuda: True; use_cuda: True
    - data_type: 'molecule'
    - batch_size: 1
    - learning_rate: 8.5e-05
    - epochs: 350; max_epochs: 100
    - patience: 60; early_stopping: 80
    - warmup_ratio: 0.03
    - kfold: 5
    - split: 'random'; split_seed: 42; split_group_col: 'scaffold'
    - smi_strict: True
    - remove_hs: False
    - smiles_col: 'SMILES'
    - target_cols: 'TARGET'; target_col_prefix: 'TARGET'; target_normalize: 'auto'
    - task: 'regression'; num_classes: 1
    - metrics: 'mse'
    - model_name: 'unimolv1'; model_size: '84m'
    - freeze_layers: 25; freeze_layers_reversed: 25
    - max_norm: 12.0
    - load_model_dir: None
    - train_from_scratch: False
    - logger_level: 1
    - seed: 42
    - anomaly_clean: True

    Returns
    - Dict[str, Path]: {"model_dir": Path(save_dir)}
    """

    merged = _default_params()
    if params:
        merged.update(params)

    save_dir_path = Path(merged["save_dir"]).resolve()
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Run training
    clf = MolTrain(**merged)
    clf.fit(str(csv_path))

    results: Dict[str, Path] = {}
    results["model_dir"] = save_dir_path
    return results


def UniMolPredict(model_dir: Path, csv_path: Path) -> List[float]:
    """
    Run inference with a UniMol model bundle (.tgz) on a CSV of molecules.

    Inputs
    - model_dir (Path): Path to a .tgz archive of the trained model.

    - csv_path (Path): Path to the input CSV for prediction. Must contain the SMILES column used during training
      (default 'SMILES'). Predictions align with the CSV row order.

    Returns
    - List[float]: List of predicted target values
    """
    logger.info(f"   开始预测: {csv_path}")
    clf = MolPredict(load_model=model_dir)
    logger.info(f"   预测模型: clf is a MolPredict object")
    y_pred = clf.predict(str(csv_path))
    logger.info(f"   预测结果: {y_pred}")

    print(f"   预测结果: {y_pred}")
    print(f"   预测结果形状: {y_pred.shape}")
    y_pred = y_pred.reshape(-1)
    return y_pred.tolist()


if __name__ == "__main__":
    import sys
    import shutil
    from pathlib import Path

    print("=== 开始单元测试 ===")

    # 测试1: SMILESListToCSV
    print("\n1. 测试 SMILESListToCSV...")
    smiles_list = ["CCO", "CCN", "C1=CC=CC=C1"]  # 乙醇、甲胺、苯
    csv_path = SMILESListToCSV(smiles_list)
    print(f"   生成的CSV文件: {csv_path}")
    print(f"   文件存在: {csv_path.exists()}")

    # 验证CSV内容
    df = pd.read_csv(csv_path)
    print(f"   CSV列: {list(df.columns)}")
    print(f"   SMILES数量: {len(df)}")
    #测试3: UniMolPredict
    print("\n3. 测试 UniMolPredict...")
    for i in range(10):
        unimol_predict_result=UniMolPredict(Path("/unimol_weight_perovskite/fixed_hyper_param"), Path("/unimol_weight_perovskite/unimol_mcp_matmaster/2025-12-02-18:13:01.882272/test/20251202_181301_smiles.csv"))
    print(f"   预测结果: {unimol_predict_result}")
    print("\n=== 单元测试完成 ===")