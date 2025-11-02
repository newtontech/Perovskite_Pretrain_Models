import os

os.environ['UNIMOL_WEIGHT_DIR'] ='<your model path here>'
save_path = '<your model path here>'


import torch
import joblib
import pandas as pd
from unimol_tools import MolTrain, MolPredict

torch.cuda.set_device(0)


csv_path = 'single_mol.csv'
current_directory = os.getcwd()
current_folder_name = os.path.basename(current_directory)

clf = MolPredict(load_model=save_path,save_heatmap=True)
clf.predict(csv_path)

