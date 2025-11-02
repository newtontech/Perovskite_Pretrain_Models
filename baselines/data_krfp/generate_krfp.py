import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from rdkit import Chem
from rdkit.Chem import MolStandardize, AllChem, Scaffolds
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optims
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class KRFPExtractor:
    def __init__(self, json_path='krfp.json'):
        with open(json_path) as f:
            self.smarts_dict = json.load(f)
        self.patterns = {
            key: Chem.MolFromSmarts(smarts) 
            for key, smarts in self.smarts_dict.items()
        }
        self.feature_names = list(self.smarts_dict.keys())
        
    def process_mol(self, mol):
        """分子标准化处理"""
        mol = MolStandardize.rdMolStandardize.Cleanup(mol)
        mol = MolStandardize.rdMolStandardize.FragmentParent(mol)
        uncharger = MolStandardize.rdMolStandardize.Uncharger()
        return uncharger.uncharge(mol)
    
    def get_krfp(self, smiles):
        """生成KRFP向量"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return np.zeros(len(self.feature_names))
        try:
            mol = self.process_mol(mol)
            return np.array([int(mol.HasSubstructMatch(pattern)) 
                   for pattern in self.patterns.values()])
        except:
            return np.zeros(len(self.feature_names))

# ----------------------
# 2. 数据加载与预处理
# ----------------------
# 初始化KRFP生成器
krfp_extractor = KRFPExtractor()


base_dir = '../data'
for i in range(5):
    file_dir = f"{base_dir}/split_seed_{i}/"
    output_file_dir =f"{base_dir}/split_seed_{i}/"
    files = ['train.csv','test.csv','test_cleaned.csv']
    for file in files:
        df = pd.read_csv(file_dir + file)
        # 生成KRFP特征
        smiles_list = df['SMILES'].tolist()
        feature_list = []
        for smi in tqdm(smiles_list, desc=f"Processing {file_dir + file}"):
            features = krfp_extractor.get_krfp(smi)
            feature_list.append(features)
        # get full feature matrix
        full_feature_matrix = np.array(feature_list)
        print("Feature matrix shape:", full_feature_matrix.shape)

        # save to file
        np.save(output_file_dir + file.replace('.csv', '_krfp.npy'), full_feature_matrix)

        print("Features saved to", output_file_dir + file.replace('.csv', '_krfp.npy'))
