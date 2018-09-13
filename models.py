import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import ConditionalAttentionLayer, UnconditionalAttentionLayer, SimplifiedGATLayer

class CCModel(nn.Module):
    """
    Model matching the form used for the CORA and Citeseer tasks in the GAT paper.
    GAT formulation:
    - 2 GAT layers
        - first has 8 heads computing 8 features each, concatenated to 64, with ELU activation
        - second has 1 head computing C features for classification with softmax
    - dropout = 0.6
    CAT modification:
    - 2 CAT layers
        - first has 8 mechanisms computing 8 features, concatenated, ELU
        - second has 1 mechanism (no conditioning) computing C features for classification with softmax
    - dropout = 0.6
    """
    def __init__(self, ins, classes, conditioner):
        super(CCModel, self).__init__()
        
        # dropout is included in the layers so we don't need to add anything else
        # activate the first layer and use the automatic ELU
        self.CAT1 = ConditionalAttentionLayer(ins=ins, outs=8, dropout=0.1, leak=0.2, N_mechs=8,
                                              conditioner=conditioner, activate=True)
        # do not activate the output
        self.CAT2 = UnconditionalAttentionLayer(N_mechs=1, dropout=0.3, ins=64, leak=0.2, outs=classes)
        
    def forward(self, x, adj):
        # pass is v tidy, just first layer then second
        x = self.CAT2((self.CAT1(x,adj)),adj)
        return F.log_softmax(x, dim=1)

class SimpleGAT(nn.Module):
    """
    Model matching the form used for the CORA and Citeseer tasks in the GAT paper, replacing the GAT
    layers with Simplified GAT Layers
    """
    def __init__(self, ins, classes):
        super(CCmodel, self).__init__()
        
        # dropout is included in the layers so we don't need to add anything else
        # activate the first layer and use the default ELU
        # layers default to concatenation 
        self.SGAT1 = SimplifiedGATLayer(N_mechs=8, dropout=0.6, ins=ins, leak=0.2, outs=8, activate=True)
        # do not activate the output
        self.SGAT2 = SimplifiedGATLayer(N_mechs=1, dropout=0.6, ins=64, leak=0.2, outs=classes)
        
    def forward(self, x, adj):
        # pass is v tidy, just first layer then second
        x = self.SGAT2((self.SGAT1(x,adj)),adj)
        return F.log_softmax(x, dim=1)