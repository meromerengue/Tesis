from torch.nn import Module
import torch
from torch.nn import ModuleList

import math
import torch.nn.functional as F
torch.cuda.empty_cache()
from Models.Decoder import Decoder
from Models.transformer_copy import Transformer

class AutoEncoder(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_hz: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):

        super(AutoEncoder, self).__init__()

        self.encoder = Transformer(d_model=d_model, 
                                    d_input=d_input, 
                                    d_channel=d_channel, 
                                    d_hz = d_hz, 
                                    d_output=d_output, 
                                    d_hidden=d_hidden, 
                                    q=q, 
                                    v=v, 
                                    h=h, 
                                    N=N, 
                                    dropout=dropout, 
                                    pe=pe, 
                                    mask=mask,
                                    device=device)
        
        self.decoder = Decoder(d_model=d_model,
                                d_input=d_input,
                                d_channel=d_channel,
                                d_hz=d_hz)
        
    def forward(self, x):

        embeddings = self.encoder(x)
        reconstruction = self.decoder(embeddings)
        return reconstruction