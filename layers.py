import torch.nn as nn
import torch
import torch.nn.functional as F
from mechanisms import ConditionalAttentionMech, SimplifiedGAT, deepCAT, unCAT

class ConditionalAttentionLayer(nn.Module):
    def __init__(self, ins, outs, dropout, leak, N_mechs, conditioner, concat=True, activate=False, activation=F.elu):
        super(ConditionalAttentionLayer, self).__init__()
        
        self.dropout = dropout
        
        # generate the N mechanisms
        self.mechanisms = [deepCAT(ins,outs,dropout,leak) for _ in range(N_mechs)]
        # add to module
        for i, mechanism in enumerate(self.mechanisms):
            self.add_module('mechanism_{}'.format(i), mechanism)
        
        self.activation = activation
        self.activate = activate
        self.concat = concat
        
        # the conditioner should take something of shape
        #     (N*F) = features (F) for each node (N)
        # and return something of shape
        #     (2*N_mechs*N) = gamma,beta (2) for each node (N) for each mechanism (N_mechs)
        self.conditioner = conditioner
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # either concatenate (intermediate layer)
        gamma, beta = self.conditioner(x)
        if self.concat:
            x = torch.cat([mech(x, adj, gamma[i], beta[i]) for i,mech in enumerate(self.mechanisms)], dim=1)
        # or sum (final layer)
        else:
            x = torch.sum([mech(x, adj, gamma[i], beta[i]) for i,mech in self.mechanisms], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.activate:
            x = self.activation(x)
        return x
    
class UnconditionalAttentionLayer(nn.Module):
    def __init__(self, ins, outs, dropout, leak, N_mechs, concat=True, activate=False, activation=F.elu):
        super(UnconditionalAttentionLayer, self).__init__()
        
        self.dropout = dropout
        
        # generate the N mechanisms
        self.mechanisms = [unCAT(ins,outs,dropout,leak) for _ in range(N_mechs)]
        # add to module
        for i, mechanism in enumerate(self.mechanisms):
            self.add_module('mechanism_{}'.format(i), mechanism)
        
        self.activation = activation
        self.activate = activate
        self.concat = concat
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # either concatenate (intermediate layer)
        if self.concat:
            x = torch.cat([mech(x, adj) for mech in self.mechanisms], dim=1)
        # or sum (final layer)
        else:
            x = sum([mech(x, adj) for mech in self.mechanisms])
        x = F.dropout(x, self.dropout, training=self.training)
        if self.activate:
            x = self.activation(x)
        return x

class SimplifiedGATLayer(nn.Module):
    def __init__(self, ins, outs, dropout, leak, N_mechs, concat=True, activate=False, activation=F.elu):
        super(SimplifiedGATLayer, self).__init__()
        
        self.dropout = dropout
        
        # generate the N mechanisms
        self.mechanisms = [SimplifiedGAT(ins,outs,dropout,leak) for _ in range(N_mechs)]
        # add to module
        for i, mechanism in enumerate(self.mechanisms):
            self.add_module('mechanism_{}'.format(i), mechanism)
        
        self.activation = activation
        self.activate = activate
        self.concat = concat
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # either concatenate (intermediate layer)
        if self.concat:
            x = torch.cat([mech(x, adj) for mech in self.mechanisms], dim=1)
        # or sum (final layer)
        else:
            x = torch.sum([mech(x, adj) for mech in self.mechanisms], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        if self.activate:
            x = self.activation(x)
        return x