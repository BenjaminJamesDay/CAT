import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Included mechanisms:
 - Conditional Attention
 - Simplified GAT
"""

class deepCAT(nn.Module):
    """
    Auto-conditional GAT-like mechanism with external conditioning:
    
    Take: h (N*F), Adj (N*N), gamma (N), beta (N)
    
    alpha_ij = a . concat(big_W . h_i, big_W . h_j) = (b . big_W . h_i) + (c . big_W . h_j)
    
    e_ij = softmax(LeakyReLU(alpha_ij))_j   -- j in neighbourhood of i
    
    GAT = sum { e_ij * big_W . h_j }
    
    h' = gamma * GAT + beta
    
    """

    def __init__(self, in_features, out_features, dropout, leak, condition=True):
        super(deepCAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.condition = condition
        self.leakyrelu = nn.LeakyReLU(leak)
        
        t_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.big_W = nn.Parameter(torch.Tensor(in_features, out_features).type(t_type), requires_grad=True)
        
        # split 'a' to better represent how it is used
        self.a_i = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        self.a_j = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        
        #initialize all with Xavier (Glorot Uniform)
        nn.init.xavier_uniform_(self.big_W, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_i, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_j, np.sqrt(2.0))
        
    def forward(self, input, adj, gamma, beta):
        # transform to new feature space
        h = torch.mm(input, self.big_W)
        # count
        N = h.size()[0]
        
        # repeat in order {1,1,...,1,2,2,...,2,3,...N}
        a_i_input = h.repeat(1, N).view(N * N, -1)
        # repeat in order {1,2...N,1,2...N,...,N}
        a_j_input = h.repeat(N, 1)
        
        # sum the dot products
        alpha = torch.matmul(a_i_input, self.a_i) + torch.matmul(a_j_input, self.a_j)
        # activate and make square
        e = self.leakyrelu(alpha).view(N,N)
        
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
            h_prime = h_prime.permute(1,0)
            # this is super easy in PyTorch where multiplication and addition are elementwise
            h_prime = (1+gamma) * h_prime + beta
            h_prime = h_prime.permute(1,0)

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class unCAT(nn.Module):
    """
    GAT-like mechanism (no conditioning)
    """

    def __init__(self, in_features, out_features, dropout, leak):
        super(unCAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(leak)
        
        t_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.big_W = nn.Parameter(torch.Tensor(in_features, out_features).type(t_type), requires_grad=True)
        
        # split 'a' to better represent how it is used
        self.a_i = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        self.a_j = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        
        #initialize all with Xavier (Glorot Uniform)
        nn.init.xavier_uniform_(self.big_W, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_i, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_j, np.sqrt(2.0))
        
    def forward(self, input, adj):
        # transform to new feature space
        h = torch.mm(input, self.big_W)
        # count
        N = h.size()[0]
        
        # repeat in order {1,1,...,1,2,2,...,2,3,...N}
        a_i_input = h.repeat(1, N).view(N * N, -1)
        # repeat in order {1,2...N,1,2...N,...,N}
        a_j_input = h.repeat(N, 1)
        
        # sum the dot products
        alpha = torch.matmul(a_i_input, self.a_i) + torch.matmul(a_j_input, self.a_j)
        # activate and make square
        e = self.leakyrelu(alpha).view(N,N)
        
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

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
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
        super(ConditionalAttentionMech, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.condition = condition
        self.leakyrelu = nn.LeakyReLU(leak)
        
        t_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.big_W = nn.Parameter(torch.Tensor(in_features, out_features).type(t_type), requires_grad=True)
        
        # split 'a' to better represent how it is used
        self.a_i = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        self.a_j = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        
        # conditioing parameter vectors
        self.w_gamma = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        self.w_beta = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        
        
        #initialize all with Xavier (Glorot Uniform)
        nn.init.xavier_uniform_(self.big_W, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_i, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a_j, np.sqrt(2.0))
        
        # lowered gain for the conditioners as we don't want them to dominate early on
        nn.init.xavier_uniform_(self.w_gamma, np.sqrt(0.005))
        nn.init.xavier_uniform_(self.w_beta, np.sqrt(0.005))


    def forward(self, input, adj):
        # transform to new feature space
        h = torch.mm(input, self.big_W)
        # count
        N = h.size()[0]
        
        # repeat in order {1,1,...,1,2,2,...,2,3,...N}
        a_i_input = h.repeat(1, N).view(N * N, -1)
        # repeat in order {1,2...N,1,2...N,...,N}
        a_j_input = h.repeat(N, 1)
        
        # sum the dot products
        alpha = torch.matmul(a_i_input, self.a_i) + torch.matmul(a_j_input, self.a_j)
        # activate and make square
        e = self.leakyrelu(alpha).view(N,N)
        
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
            gamma = torch.matmul(h, self.w_gamma)
            gamma += torch.ones_like(gamma)
            beta = torch.matmul(h, self.w_beta)
            
            h_prime = gamma * h_prime + beta

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    
class SimplifiedGAT(nn.Module):
    """
    Simplified GAT-like mechanism dropping h_i dependency in the attention mechanism entirely
    on the basis that it was only acting as a severity check and could not change the rankings
    of neighbours
    
    alpha_ij = a . concat(big_W . h_j)
    
    e_ij = softmax(LeakyReLU(alpha_ij))_j   -- j in neighbourhood of i
    
    h' = sum { e_ij * big_W . h_j }
    """

    def __init__(self, in_features, out_features, dropout, leak):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(leak)
        
        t_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.big_W = nn.Parameter(torch.Tensor(in_features, out_features).type(t_type), requires_grad=True)
        
        # this a is only used for neighbours (hence only out_features*1)
        self.a = nn.Parameter(torch.Tensor(out_features, 1).type(t_type), requires_grad=True)
        
        #initialize all with Xavier (Glorot Uniform)
        nn.init.xavier_uniform_(self.big_W, np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a, np.sqrt(2.0))


    def forward(self, input, adj):
        # transform to new feature space
        h = torch.mm(input, self.big_W)
        # count
        N = h.size()[0]
        
        # repeat in order {1,2...N,1,2...N,...,N}
        a_input = h.repeat(N, 1)
        
        # take dot product with a
        alpha = torch.matmul(a_input, self.a)
        # activate and form into a square matrix
        e = self.leakyrelu(alpha, leak).view(N,N)
        
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

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'