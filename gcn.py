import torch
import torch.nn as nn


class ConvGraph(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.FloatTensor(size_out, size_in))
        # self.bias = nn.Parameter(torch.Tensor())

    def forward(self, x, A):
        x = torch.mm(x, self.weights)
        x = torch.spmm(A, x)
        return x
