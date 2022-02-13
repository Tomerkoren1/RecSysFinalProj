#networks.py
#Modified by ImKe on 2019/9/4.
#Copyright © 2019 ImKe. All rights reserved.

from collections import OrderedDict
import torch
from torch import nn

def activation(type):
  
    if type.lower()=='selu':
        return nn.SELU()
    elif type.lower()=='elu':
        return nn.ELU()
    elif type.lower()=='relu':
        return nn.ReLU()
    elif type.lower()=='relu6':
        return nn.ReLU6()
    elif type.lower()=='lrelu':
        return nn.LeakyReLU()
    elif type.lower()=='tanh':
        return nn.Tanh()
    elif type.lower()=='sigmoid':
        return nn.Sigmoid()
    elif type.lower()=='identity':
        return nn.Identity()
    else:
        raise ValueError("Unknown non-Linearity Type")

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super(AutoEncoder, self).__init__()
        d1 = OrderedDict()
        for i in range(len(hidden)-1):
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])#nn.Linear(input,out,bias=True)
            #d1['enc_bn' + str(i)] = nn.BatchNorm1d(hidden[i + 1])
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            d1['enc_relu'+str(i)] = nn.Sigmoid()
        self.encoder = nn.Sequential(d1)
        d2 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d2['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            #d2['dec_bn' + str(i)] = nn.BatchNorm1d(hidden[i - 1])
            d2['dec_drop' + str(i)] = nn.Dropout(dropout)
            d2['dec_relu' + str(i)] = nn.Sigmoid()
        self.decoder = nn.Sequential(d2)

    def forward(self, x):
        #norm1d
        x = (x-1)/5.0
        x = self.decoder(self.encoder(x))
        x = torch.clamp(x, 0, 1.0)#torch.clamp(input, min, max)
        x = x * 5.0 + 1
        return x

def init_params(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.uniform_(layer.bias)

class OurAutoEncoder(nn.Module):
    def __init__(self, hidden, act_type, dropout=0.1):
        super(OurAutoEncoder, self).__init__()
        d1 = OrderedDict()
        for i in range(len(hidden)-1):
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])#nn.Linear(input,out,bias=True)
            init_params(d1['enc_linear' + str(i)])
            # d1['enc_bn' + str(i)] = nn.BatchNorm1d(hidden[i + 1])
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            d1['enc_relu'+str(i)] = activation(act_type)
        self.encoder = nn.Sequential(d1)
        d2 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d2['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            init_params(d2['dec_linear' + str(i)])
            # d2['dec_bn' + str(i)] = nn.BatchNorm1d(hidden[i - 1])
            d2['dec_drop' + str(i)] = nn.Dropout(dropout)
            d2['dec_relu' + str(i)] = activation(act_type)
            
            
        self.decoder = nn.Sequential(d2)

    def forward(self, x):
        #norm1d
        x = (x-1)/5.0
        x = self.decoder(self.encoder(x))
        x = torch.clamp(x, 0, 1.0)#torch.clamp(input, min, max)
        x = x * 5.0 + 1
        return x