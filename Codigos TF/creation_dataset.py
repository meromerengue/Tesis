import pandas as pd 
import numpy as np


data_csv_EEG = pd.read_csv("/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/new_spectro_ictal.csv")
new = data_csv_EEG.copy()


#test_15_1 = data_csv_EEG[data_csv_EEG['label'].isin([1])]

#print(test_15_1.shape)

#test_15 = data_csv_EEG[data_csv_EEG['Paciente'].isin([17,21,22,15])]
test_15 = data_csv_EEG[data_csv_EEG['Paciente'].isin([2, 3, 4])]
print(test_15.info())
train_15 = data_csv_EEG[data_csv_EEG['Paciente'].isin([1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17])]
#train_15 = data_csv_EEG[data_csv_EEG['Paciente'].isin([1,2,3,4,5,6,7,9,11,14,21,22,16,18,9])]
print(train_15.info())

test_15.to_csv('/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_test_1_ictal.csv', index=False)
train_15.to_csv('/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_train_1_ictal.csv', index=False)



