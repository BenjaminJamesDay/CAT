import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalAttentionMech(nn.Module):
    """
    Auto-conditional GAT-like mechanism:
    
    alpha_ij = a . concat(big_W . h_i, big_W . h_j) = (b . big_W . h_i) + (c . big_W . h_j)
    
    e_ij = softmax(LeakyReLU(alpha_ij))_j   -- j in neighbourhood of i
    
    GAT = sum { e_ij * big_W . h_j }
    
    gamma = w_g . h' = w_g . big_W . h
    beta = w_b . h' = w_b . big_W . h
    
    h' = gamma * GAT + beta
    
    """

    def __init__(self, in_features, out_features, dropout, leak, condition=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.leak = alpha
        self.condition = condition
        
        t_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.big_W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(t_type), 
                                                         gain=np.sqrt(2.0)), 
                                  requires_grad=True)
        
        # split 'a' to better represent how it is used
        self.a_i = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(t_type),
                                                       gain=np.sqrt(2.0)),
                                requires_grad=True)
        self.a_j = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(t_type),
                                                       gain=np.sqrt(2.0)),
                                requires_grad=True)
        
        # conditioning parameter vectors
        self.w_gamma = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(t_type),
                                                           gain=np.sqrt(2.0)),
                                    requires_grad=True)
        
        self.w_beta = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(t_type),
                                                          gain=np.sqrt(2.0)),
                                   requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.leak)

    def forward(self, input, adj):
        # transform to new feature space
        h = torch.mm(input, self.big_W)
        # count
        N = h.size()[0]
        
        # repeat in order {1,1,...,1,2,2,...,2,3,...N}
        a_i_input = h.repeat(1, N).view(N * N, -1)
        # repeat in order {1,2...N,1,2...N,...,N}
        a_j_input = h.repeat(N, 1)
        
        # sum the two dot products
        alpha = torch.sum(torch.matmul(a_i_input, self.a_i), torch.matmul(a_j_input, self.a_j))
        # activate
        e = self.leakyrelu(alpha, leak)
        
        # this will zero out in a softmax
        zero_vec = -9e15*torch.ones_like(e)
        # mask using adjacency matrix
        attention = torch.where(adj > 0, e, zero_vec)
        # take softmax
        attention = F.softmax(attention, dim=1)
        # perform dropout (not sure why they do this here and not on the adj but w/e)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # give new features
        h_prime = torch.matmul(attention, h)
        
        # if this is a conditioning layer, do the conditioning
        if self.condition:
            gamma = torch.matmul(a_i_input, self.w_gamma)
            gamma += torch.ones_like(gamma)
            beta = torch.matmul(a_i_input, self.w_beta)
            
            h_prime = gamma * h_prime + beta

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'