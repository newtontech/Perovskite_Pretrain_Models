import os

os.environ['UNIMOL_WEIGHT_DIR'] ='<your model path here>'
save_path = '<your model path here>'
output_dir = 'addition_points_random_features_trained_model.pt'

import torch
import joblib
import pandas as pd
from unimol_tools import MolTrain, MolPredict

torch.cuda.set_device(0)


csv_path = '10_mol.csv'
current_directory = os.getcwd()
current_folder_name = os.path.basename(current_directory)

clf = MolPredict(load_model=save_path, random_weight=True)
prediction,features = clf.predict(csv_path)
print("Saving features of size ", features[0].shape, "to", output_dir)
torch.save(features[0],output_dir)


