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
        self.hidden3 = nn.Linear(32, outs)
        
        nn.init.xavier_uniform_(self.hidden1.weight, 0.05)
        nn.init.xavier_uniform_(self.hidden2.weight, 0.05)
        nn.init.xavier_uniform_(self.hidden3.weight, 0.05)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return (self.hidden3(x))