import torch
import torch.nn as nn
import math

class ConvGraph(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.FloatTensor(size_out, size_in))
        # self.bias = nn.Parameter(torch.Tensor())
        self.init_weights()


    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        x = torch.mm(x, self.weights)
        x = torch.spmm(A, x)
        return x
