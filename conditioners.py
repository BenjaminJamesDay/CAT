import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Included conditioners:
 - 3 layer ANN for Cora 2708*(1433->32->32->16)
 - 3 layer ANN for 
"""

class coraConditioner(nn.Module):
    def __init__(self, out_params):
        super(coraConditioner, self).__init__()
        self.hidden1 = nn.Linear(1433,64)
        self.hidden2 = nn.Linear(64,32)
        self.hidden3 = nn.Linear(32, out_params)
        
        nn.init.xavier_uniform_(self.hidden1.weight, 1)
        nn.init.xavier_uniform_(self.hidden2.weight, 1)
        nn.init.xavier_uniform_(self.hidden3.weight, 1)

    def forward(self, x):
        x = F.elu(self.hidden1(x))
        x = F.dropout(x, 0.7, training=self.training)
        x = F.elu(self.hidden2(x))
        x = F.dropout(x, 0.7, training=self.training)
        return (self.hidden3(x))