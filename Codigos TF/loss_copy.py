from torch.nn import Module
import torch
from torch.nn import MSELoss, Module

class MyMSE(Module):
    def __init__(self):
        super(MyMSE, self).__init__()
        self.loss_function = MSELoss()

    def forward(self, y_pre, y_true):
        # Asegurarse de que las entradas sean de tipo float para MSELoss
        y_pre = y_pre.float()
        y_true = y_true.float()
        
        loss = self.loss_function(y_pre, y_true)
        return loss