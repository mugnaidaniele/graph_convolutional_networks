import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ConvGraph(nn.Module):
    def __init__(self, size_in, size_out, dropout=0.5):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.FloatTensor(size_in, size_out))
        # self.bias = nn.Parameter(torch.Tensor())
        self.init_weights()
        self.dropout = dropout

    def init_weights(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        # print(x.shape, self.weights.shape )
        x = torch.mm(x, self.weights)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(A, x)
        return x
