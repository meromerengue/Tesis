from torch.nn import Module
import torch
from torch.nn import MSELoss, Module

class MyMaskedMSE(Module):
    def __init__(self):
        super(MyMaskedMSE, self).__init__()
        self.loss_function = MSELoss()

    def forward(self, y_pre, y_true, mask):
        # Asegurarse de que las entradas sean de tipo float para MSELoss
        y_pre_masked = y_pre.float() * mask
        y_true_masked = y_true.float() * mask
        pre_loss = self.loss_function(y_pre_masked, y_true_masked)
        total_pixels = torch.prod(torch.tensor(y_true.shape))
        mask_pixels = mask.sum()
        loss = pre_loss * (total_pixels/mask_pixels)
        return loss