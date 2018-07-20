import torch.nn as nn
import torch
import torch.nn.functional as F
from layer import ConditionalAttentionLayer

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
    def __init__(self, ins, classes):
        super(CCmodel, self).__init__()
        
        # a leak of 1 linearises the layers so we can use ELU and softmax instead
        # dropout is included in the layers
        self.CAT1 = ConditionalAttentionLayer(N_mechs=8, concat=True, dropout=0.6, ins=ins, leak=1, outs=8)
        self.CAT2 = ConditionalAttentionLayer(N_mechs=1, cocnat=True, dropout=0.6, ins=64, leak=1, outs=classes)
        
    def forward(self, x, adj):
        # pass is v tidy, just first layer then second
        x = F.elu(self.CAT1(x,adj))
        return F.log_softmax(x, dim=1)

