import csv
import numpy as np
import pandas  as pd
column_data = []
split_nums = range(10)
base_dir = '/data/yhm/cxy_molclr/MolCLR/data/perovskite-resplit/split_seed_4/'
file_names = ['test_cleaned_with_predictions_formal.csv','train_with_predictions_formal.csv','test_with_predictions_formal.csv']
raw_file_names = ['test_cleaned.csv','train.csv','test.csv']

for i,file_name in enumerate(file_names):
    for split_num in split_nums:
        file_path = base_dir + file_name.split('.')[0] + '_' + str(split_num) + '.csv'
        predictions = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)  
            prediction = []
            for row in reader:
                prediction.append(float(row['prediction']))
            predictions.append(np.array(prediction))
    print(np.mean(predictions, axis=0))
    raw_file_name = raw_file_names[i]
    path = base_dir+raw_file_name.replace('.csv', '_ensemble_new.csv')
    raw_df = pd.read_csv(base_dir+raw_file_name)
    raw_df['prediction'] =  np.mean(predictions, axis=0)
    raw_df.to_csv(path, index=False)






