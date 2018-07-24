import torch.nn as nn
import torch
import torch.nn.functional as F
from mechanism import ConditionalAttentionMech

class ConditionalAttentionLayer(nn.Module):
    def __init__(self, ins, outs, dropout, leak, N_mechs, concat=True):
        super(ConditionalAttentionLayer, self).__init__()
        
        self.dropout = dropout
        
        # generate the N mechanisms
        self.mechanisms = [ConditionalAttentionMech(ins,outs,dropout,leak) for _ in range(N_mechs)]
        # add to module
        for i, mechanism in enumerate(self.mechanisms):
            self.add_module('mechanism_{}'.format(i), mechanism)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # either concatenate (intermediate layer)
        if concat:
            x = torch.cat([mech(x, adj) for mech in self.mechanisms], dim=1)
        # or sum (final layer)
        else:
            x = torch.sum([mech(x, adj) for mech in self.mechanisms], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x