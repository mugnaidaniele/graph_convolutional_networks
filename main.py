from model import GCN
import torch
net = GCN(10, 2, 4)
A = torch.rand((10,10))
i = torch.rand(10)
print(i.size())
print(i)
o = net(i, A)