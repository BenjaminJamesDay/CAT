import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Included conditioners:
 - 3 layer ANN for Cora 2708*(1433->32->32->16)
 - 3 layer ANN for 
"""

class coraConditionerModel(nn.Module):
    def __init__(self):
        super(coraConditionerModel, self).__init__()
        self.hidden1 = nn.Linear(1433,64)
        self.hidden2 = nn.Linear(64,32)
        self.hidden3 = nn.Linear(32,36)
        
        nn.init.xavier_uniform_(self.hidden3.weight, 0.05)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return (self.hidden3(x))

def coraConditioner(x):
    """
    Input is the Cora dataset 2708*1433
    For each node (2708) the features (1433) are used to generate conditioning
    parameters (16 = 8*2) in parallel. That is, the same network is used to
    transform the features to the conditioning parameters for every node. Thus
    the network itself is 1433->..->16
    """
    
    model = coraConditionerModel()
    model.cuda()
    cond = torch.cat([model(example) for example in x])
    #reshape the concatenated set (2N.N_mechs) to (N*2*N_mechs) then permute to (2*N_mechs*N)
    return cond.view(2708,2,18).permute(1,2,0)

class coraConditionerModel2(nn.Module):
    def __init__(self):
        super(coraConditionerModel2, self).__init__()
        self.hidden1 = nn.Linear(1433,64)
        self.hidden2 = nn.Linear(64,32)
        self.hidden3 = nn.Linear(32,32)
        
        nn.init.xavier_uniform_(self.hidden3.weight, 0.05)

    def forward(self, x):
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        return (self.hidden3(x))

def coraConditioner2(x):
    """
    Input is the Cora dataset 2708*1433
    For each node (2708) the features (1433) are used to generate conditioning
    parameters (16 = 8*2) in parallel. That is, the same network is used to
    transform the features to the conditioning parameters for every node. Thus
    the network itself is 1433->..->16
    """
    
    model = coraConditionerModel2()
    model.cuda()
    cond = torch.cat([model(example) for example in x])
    #reshape the concatenated set (2N.N_mechs) to (N*2*N_mechs) then permute to (2*N_mechs*N)
    return cond.view(2708,2,16).permute(1,2,0)