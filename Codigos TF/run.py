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




import h5py
import wandb
from loss import Myloss
from Models.transformer import Transformer
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch

# HDF5_USE_FILE_LOCKING='FALSE'

# os.environ["WANDB_DIR"]= "/home/mreyes/"

# os.environ["WANDB_CACHE_DIR"]= "/home/mreyes/"

# os.environ["WANDB_CONFIG_DIR"]= "/home/mreyes/"

# wandb.login(key="0c61fa16d8327be469855b85b503241498dd4f44")

# wandb.init(
#     project="Memoria"
# )


test_interval = 100  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像

# 超参数设置
EPOCH = 1
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
    
#train = EEGLandmarksDataset(csv_file='/home/scallejas/Memoria/EEG_spectro/train_nan.csv', root_dir='/home/scallejas/Memoria/EEG_spectro/EEG_segments.h5')
#test = EEGLandmarksDataset(csv_file='/home/scallejas/Memoria/EEG_spectro/test_nan.csv', root_dir='/home/scallejas/Memoria/EEG_spectro/EEG_segments.h5')
train = EEGLandmarksDataset(csv_file='/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_train_1_ictal.csv', root_dir='/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/new_EEG_segments_ictal.h5')
test = EEGLandmarksDataset(csv_file='/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/time_test_1_ictal.csv', root_dir='/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/new_EEG_segments_ictal.h5')


train_dataloader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)

#print(test[0]["x_chunk"])
inputs = train[0]["x_chunk"].shape[0]

channels = train[0]["x_chunk"].shape[1] # 时间序列维度

outputs = 2 # 分类类别
hz = train[0]["x_chunk"].shape[2]  # hz

#net = Transformer(l=64,f=8,g=16, d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens,q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).cuda()

net = Transformer(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens, q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE)
#net = torch.nn.DataParallel(net)
net = net.cuda()

# 创建Transformer模型

# 创建loss函数 此处使用 交叉熵损失
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
sensitivity_on_train = []
sensitivity_on_test = []
specificity_on_train = []
specificity_on_test = []
precision_on_train = []
precision_on_test = []
recall_on_train = []
recall_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0
net.train()
max_accuracy = 0
max_sensitivity = 0
max_specificity = 0
max_precision = 0
max_recall = 0





def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    loss_temp = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        net.eval()
        TP=0
        TN=0
        FN=0
        FP=0
        for batch in dataloader:
            x = batch["x_chunk"].cuda()
            y = batch["y_chunk"].cuda()

            y_pre = net(x,"test")
            loss = loss_function(y_pre, y)
            loss_temp.append(loss.item())

            _, label_index = torch.max(y_pre, dim=1)
            
            total += y.size(0)
            correct += (label_index == y).sum().item()

            y_true.extend(y.cpu().numpy())
            
            y_pred.extend(label_index.cpu().numpy())

        accuracy = 100 * correct / total
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        print(confusion_matrix(y_true, y_pred))
        #wandb.log({"test_accuracy": accuracy , "test_loss":sum(loss_temp)/len(loss_temp),"test_f1":f1,"test_recall":recall,"test_precision":precision})
        print({"test_accuracy": accuracy , "test_loss":sum(loss_temp)/len(loss_temp),"test_f1":f1,"test_recall":recall,"test_precision":precision})







def train():
    for index in range(EPOCH):
        loss_temp = []
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        for batch in train_dataloader:

            
            x = batch["x_chunk"].cuda()
            y = batch["y_chunk"].cuda()
            print(f"dimension de x: {x.shape}")
            optimizer.zero_grad()

            y_pre = net(x,"train")
            loss = loss_function(y_pre, y)
            print(f"dimension de la salida: {y.shape}")
            

            loss_temp.append(loss.item())
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

            predicted = torch.argmax(y_pre, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            y_true.extend(y.cpu().numpy())
            
            y_pred.extend(predicted.cpu().numpy())

        #torch.save(net, f'saved_model/{"modelo_spectro"} batch={BATCH_SIZE}.pkl')

        accuracy = 100 * correct / total
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        #wandb.log({"train_accuracy": accuracy , "train_loss":sum(loss_temp)/len(loss_temp),"train_f1":f1,"train_recall":recall,"train_precision":precision})
        print({"train_accuracy": accuracy , "train_loss":sum(loss_temp)/len(loss_temp),"train_f1":f1,"train_recall":recall,"train_precision":precision})
        torch.save(net.state_dict(), '/home/mreyes/Documents/Git/Tesis/EEG_spectro_frec/spectro_1.pth')



        print(f'Epoch:{index + 1}:\t\tloss:{sum(loss_temp)/len(loss_temp)}')
        
        test(test_dataloader)


if __name__ == '__main__':
    train()