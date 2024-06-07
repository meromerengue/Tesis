from torch.nn import Module
import torch
from torch.nn import ModuleList

import math
import torch.nn.functional as F
torch.cuda.empty_cache()

from Models.Encoder import Encoder
class Decoder(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_hz: int):

        super(Decoder, self).__init__()

        self.d_input = d_input
        self.d_channel = d_channel
        self.d_hz = d_hz
        
        self.Layer_1 = torch.nn.Linear(d_input + d_channel + d_hz, d_input*d_channel*d_hz)
        self.Layer_2 = torch.nn.Linear(d_model, 1)

    def forward(self, x):

        decode_1 = self.Layer_1(x.permute(0, 2, 1))
        decode_2 = self.Layer_2(decode_1.permute(0, 2, 1))
    
        output = decode_2.view(decode_2.shape[0], self.d_input, self.d_channel, self.d_hz)

        return output