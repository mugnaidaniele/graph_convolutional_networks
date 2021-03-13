import torch
import torch.nn as nn
from torch.functional import F
from gcn import ConvGraph

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(GCN, self).__init__()
        self.conv1 = ConvGraph(input_size, hidden_size)
        self.conv2 = ConvGraph(hidden_size, n_classes)

    def forward(self, x, A):
        x = F.relu(self.conv1(x, A))
        x = self.conv2(x, A)
        return F.log_softmax(x, dim=1)
