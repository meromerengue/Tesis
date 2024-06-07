import os
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import pandas as pd
import stft
import h5py
from sklearn.covariance import ledoit_wolf
import pyriemann
from sklearn.model_selection import cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline

# DATASET: https://physionet.org/pn6/chbmit/
sampleRate = 256
pathDataSet = '/home/scallejas/Memoria/physionet.org/files/chbmit/1.0.0'# path of the dataset
FirstPartPathOutput1='/home/scallejas/Memoria/EEG_segments'#path where the spectogram will be saved
FirstPartPathOutput='/home/scallejas/Memoria/EEG_spectro'#path where the segments will be saved

patients = ["01","02","03","04","05","06","07","08","09","10","11","13","14","15","16","17","18","19","21","22"]
#patients = ["01","02","03","05","09","11","14","17","21","22"]
#patients = ["01"]
channels=18

_30_MINUTES_OF_DATA = 256*60*30
_MINUTES_OF_DATA_BETWEEN_PRE_AND_SEIZURE = 3#SPH
_MINUTES_OF_PREICTAL = 3#SOPcd
_SIZE_WINDOW_IN_SECONDS = 5
_SIZE_WINDOW_SPECTOGRAM = _SIZE_WINDOW_IN_SECONDS*256

#para el dataframe
paciente = []
label= []
id1 =[]
id2 =[]


nSpectogram=0
signalsBlock=None
SecondPartPathOutput=''
legendOfOutput=''
isPreictal=''



def loadParametersFromFile(filePath):
    global pathDataSet
    global FirstPartPathOutput1
    global FirstPartPathOutput
    if(os.path.isfile(filePath)):
        with open(filePath, "r") as f:
                line=f.readline()
                if(line.split(":")[0]=="pathDataSet"):
                    pathDataSet=line.split(":")[1].strip()
                line=f.readline()
                if (line.split(":")[0] == "FirstPartPathOutput"):
                    FirstPartPathOutput = line.split(":")[1].strip()
                line = f.readline()
                if (line.split(":")[0] == "FirstPartPathOutput1"):
                    FirstPartPathOutput1 = line.split(":")[1].strip()

def saveSignalsOnDisk(signalsBlock,nSpectogram):
    global SecondPartPathOutput
    global FirstPartPathOutput
    global legendOfOutput
    global isPreictal
    for i,j in zip(range(0,signalsBlock.shape[0]),range(nSpectogram-signalsBlock.shape[0],nSpectogram)):
        paciente.append(SecondPartPathOutput[8:])
        label.append(isPreictal)
        id1.append(j)
        with h5py.File('/home/scallejas/Memoria/EEG_spectro/EEG_segments_R.h5', 'a') as archivo:
            archivo.create_dataset(SecondPartPathOutput[8:] + "_" + isPreictal + "_" + str(j), data=signalsBlock[i,:,:])
    
    
    
    
    
    if not os.path.exists(FirstPartPathOutput):
        os.makedirs(FirstPartPathOutput)
    if not os.path.exists(FirstPartPathOutput+SecondPartPathOutput):
        print(SecondPartPathOutput)
        os.makedirs(FirstPartPathOutput+SecondPartPathOutput)
    
    np.save(FirstPartPathOutput+SecondPartPathOutput+'/'+isPreictal+'_'+str(nSpectogram-signalsBlock.shape[0])+'_'+str(nSpectogram-1), signalsBlock)
    legendOfOutput=legendOfOutput+str(nSpectogram-signalsBlock.shape[0])+' '+str(nSpectogram-1) +' '+SecondPartPathOutput+'/'+isPreictal+'_'+str(nSpectogram-signalsBlock.shape[0])+'_'+str(nSpectogram-1) +'.npy\n'



def createSpectrogram(data, S=0):
    global nSpectogram
    global signalsBlock
    global inB
    signals = np.zeros((1, 171))

    t = 0
    movement = int(S * 256)
    if (S == 0):
        movement = _SIZE_WINDOW_SPECTOGRAM
    while data.shape[1] - (t * movement + _SIZE_WINDOW_SPECTOGRAM) >= 0:
        # 为所有通道创建光谱图

        start = t * movement
        stop = start + _SIZE_WINDOW_SPECTOGRAM
        signals = createSpec(np.expand_dims(data[:,start:stop], axis=0))
        if (signalsBlock is None):
            signalsBlock = np.array([signals])
        else:
            signalsBlock = np.append(signalsBlock, [signals], axis=0)
        nSpectogram = nSpectogram + 1
        if (signalsBlock.shape[0] == 100):
            saveSignalsOnDisk(signalsBlock, nSpectogram)
            signalsBlock = None
        
        nSpectogram = nSpectogram + 1
        t = t + 1
    return (data.shape[1] - t * _SIZE_WINDOW_SPECTOGRAM)

# 带阻滤波器
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

# 高通滤波器
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y


# 用于实际创建频谱图的函数。
def createSpec(data):
    fs = 256
    lowcut = 117
    highcut = 123

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    covest = Covariances('oas')
    ts = TangentSpace()

    clf = make_pipeline(covest, ts)
    
    Xt = clf.fit_transform(y)

    return Xt


def loadDataPath(indexPat):
    global interictalpath
    global preictalpath
    global nSeizure
    global contI
    global contP
    interictalpath = []
    preictalpath = []

    f = open(FirstPartPathOutput1 + '/patient' + patients[indexPat] + '/datamenu.txt', 'r')
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()


    contI = -1
    while ("npy" in line):
        interictalpath.append([])
        contI = contI + 1
        interictalpath[contI].append(line.rstrip('\n'))  # .rstrip() remove \n
        line = f.readline()
    line = f.readline()  # PREICTAL
    line = f.readline()

    contP = -1
    while ("npy" in line):
        preictalpath.append([])
        contP = contP + 1
        preictalpath[contP].append(line.rstrip('\n'))
        line = f.readline()
    f.close()

def main():
    global SecondPartPathOutput
    global FirstPartPathOutput1
    global legendOfOutput
    global nSpectogram
    global signalsBlock
    global isPreictal
    global interictalpath
    global preictalpath
    global contI
    global contP

    loadParametersFromFile("5sSpectogram.txt")
    for indexPatient in range(0, len(patients)):
        print("Working on patient " + patients[indexPatient])
        legendOfOutput = ""
        allLegend = ""
        nSpectogram = 0
        SecondPartPathOutput = '/patient' + patients[indexPatient]
        loadDataPath(indexPatient)
        print("START creation interictal spectrogram")
        totInst = 0
        isPreictal = 'I'  # 发作间期

        for i in range(0, contI + 1):
            
            filesPathI = interictalpath[i]
            
            
            interictaldata = np.load(FirstPartPathOutput1 + filesPathI[0])
            
            notUsed = createSpectrogram(interictaldata)
            
            totInst += interictaldata.shape[1] / 256 - notUsed / 256
            
            interictaldata = np.delete(interictaldata, np.s_[0:interictaldata.shape[1] - notUsed], axis=1)
            
        S = (_SIZE_WINDOW_IN_SECONDS * (
                    (contP + 1) * _MINUTES_OF_PREICTAL * 60 - _SIZE_WINDOW_IN_SECONDS * (contP + 1))) / totInst
        
        if (not (signalsBlock is None)):
            saveSignalsOnDisk(signalsBlock, nSpectogram)
        signalsBlock = None
        legendOfOutput = str(nSpectogram) + "\n" + legendOfOutput
        legendOfOutput = "INTERICTAL" + "\n" + legendOfOutput
        legendOfOutput = "SEIZURE: " + str(contP + 1) + "\n" + legendOfOutput
        legendOfOutput = patients[indexPatient] + "\n" + legendOfOutput
        
        allLegend = legendOfOutput
        print(legendOfOutput)
        legendOfOutput = ''
        print("END create interictal data")
        nSpectogram = 0
        isPreictal = 'P' 

        for i in range(0, contP + 1):
            legendOfOutput = legendOfOutput + "SEIZURE " + str(i) + "\n"
            filesPathP = preictalpath[i]
            preictaldata = np.load(FirstPartPathOutput1 + filesPathP[0])

            notUsed = createSpectrogram(preictaldata, S=S)

        if (not (signalsBlock is None)):
            saveSignalsOnDisk(signalsBlock, nSpectogram)
        signalsBlock = None
        allLegend = allLegend + "\n" + "PREICTAL" + "\n" + str(nSpectogram) + "\n" + legendOfOutput
        print(legendOfOutput)
        print("Spectrogram preictal: " + str(nSpectogram))
        print("SEIZURE: " + str(contP + 1))


        text_file = open(FirstPartPathOutput + SecondPartPathOutput + "/datamenu.txt", "w")
        text_file.write(allLegend)
        text_file.close()
        print("Legend saved on disk")
        print('\n')

        data = {
            'Paciente': paciente,
            'Label': label,
            'id1': id1,
            }

        df = pd.DataFrame(data)

        df.to_csv('/home/scallejas/Memoria/EEG_spectro/R_Segments.csv', index=False)


if __name__ == '__main__':
    main()

