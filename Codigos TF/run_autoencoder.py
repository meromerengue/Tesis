import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
import tqdm
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm



import h5py
import wandb
from loss_copy import MyMSE
from MaskedMSE import MyMaskedMSE
from Models.Autoencoder_features import AutoEncoder
from Models.transformer_copy import Transformer
from Models.Decoder import Decoder
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

test_interval = 100  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像

# 超参数设置
EPOCH = 24
BATCH_SIZE = 32

LR = 1e-3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = 512
hiddens = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True
mask = True  

optimizer_name = 'Adagrad'

class EEGLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_csv_EEG = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_csv_EEG)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        with h5py.File(self.root_dir, 'r') as archivo_hdf5:

            if int(self.data_csv_EEG["Paciente"][idx])/10 < 1:
                Paciente = "0" + str(self.data_csv_EEG["Paciente"][idx])
            else:
                Paciente = str(self.data_csv_EEG["Paciente"][idx])
            
            dataset = str(Paciente) + "_" + str(self.data_csv_EEG["Label"][idx]) + "_" + str(self.data_csv_EEG["id1"][idx]) 
            data = archivo_hdf5[dataset][:]
            #print(archivo_hdf5[dataset].shape)
        
        x_chunk = np.array(data)
        x_chunk = torch.Tensor(x_chunk).permute(1, 0, 2)
        x_chunk = torch.Tensor(x_chunk)
        
        if self.data_csv_EEG["Label"][idx] == "I":
            y_chunk = 0
        else:
            y_chunk = 1

        
        sample = {"x_chunk": x_chunk, "y_chunk": y_chunk}

        if self.transform:
            sample = self.transform(sample)

        return sample

def masking(batch, prob):
    mask = (torch.rand_like(batch) < prob).float()
    neg_mask = 1-mask
    masked_images = (batch*neg_mask) -(10*mask)  
    return mask, masked_images

csv_train = '/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_train_1_ictal.csv'
csv_test = '/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_test_1_ictal.csv'
path_h5 = '/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/new_EEG_segments_ictal.h5'

train = EEGLandmarksDataset(csv_file=csv_train, root_dir=path_h5)
test = EEGLandmarksDataset(csv_file=csv_test, root_dir=path_h5)


train_dataloader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)

inputs = train[0]["x_chunk"].shape[0]

channels = train[0]["x_chunk"].shape[1]

outputs = 2
hz = train[0]["x_chunk"].shape[2] 


stage = 'finetune'
print(stage)

net = AutoEncoder(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens, q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE)

if stage == 'finetune':
    pretrain_model = '/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/spectro_1_raw_train.pth'
    net.load_state_dict(torch.load(pretrain_model))

net = net.cuda()



if stage == 'finetune':
    loss_function = MyMaskedMSE()
elif stage == 'raw_train':
    loss_function = MyMSE()

if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

prob_masking = 0.15
loss_list = []


def train():
    best_loss = float('inf')
    patience = 5
    for index in tqdm(range(EPOCH)):
        loss_temp_train = []
        for batch in train_dataloader:

            
            x_real = batch["x_chunk"].cuda()
            mask, x_masked = masking(x_real, prob_masking)
            x_masked = x_masked.cuda()
            mask = mask.cuda()
            optimizer.zero_grad()

            x_recons = net(x_masked)

            if stage == 'finetune':
                loss = loss_function(x_recons, x_real, mask)
            elif stage == 'raw_train':
                loss = loss_function(x_recons, x_real)
            

            loss_temp_train.append(loss.item())
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()
        #print( f"train_loss: {sum(loss_temp_train)/len(loss_temp_train)}")
        torch.save(net.state_dict(), f'/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/spectro_1_'+stage+'.pth')
        print(f'Epoch:{index + 1}:\t\ttrain_loss:{sum(loss_temp_train)/len(loss_temp_train)}')
        
        loss_temp_test = []
        with torch.no_grad():
            net.eval()
            for batch in test_dataloader:
                x_real = batch["x_chunk"].cuda()
                mask, x_masked = masking(x_real, prob_masking)
                x_masked = x_masked.cuda()
                mask = mask.cuda()
                x_recons = net(x_masked)

                if stage == 'finetune':
                    loss = loss_function(x_recons, x_real, mask)
                elif stage == 'raw_train':
                    loss = loss_function(x_recons, x_real)
                
                loss_temp_test.append(loss.item())

            print(f"test_loss: {sum(loss_temp_test)/len(loss_temp_test)}")
            test_loss = sum(loss_temp_test)/len(loss_temp_test)
            if  test_loss < best_loss:
                best_loss = test_loss
                best_model_weights = net.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print('Early stopping!')
                torch.save(best_model_weights, f'/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/spectro_1_'+stage+'.pth')
                break
        #torch.save(net, f'saved_model/{"modelo_spectro"} batch={BATCH_SIZE}.pkl')
    

        #wandb.log({"train_accuracy": accuracy , "train_loss":sum(loss_temp)/len(loss_temp),"train_f1":f1,"train_recall":recall,"train_precision":precision})
        
        


if __name__ == '__main__':
    train()

