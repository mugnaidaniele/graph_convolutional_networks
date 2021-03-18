import time
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from data_utils import load_data, accuracy, load_citeseer
from model import GCN

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_citeseer()
# Model and optimizer
model = GCN(input_size=features.shape[1],
            hidden_size=args.hidden,
            n_classes=labels.max().item() + 1)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

criterion = nn.CrossEntropyLoss()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = criterion(output[idx_train], labels[idx_train]) #only labeled
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)

    loss_val = criterion(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val


def test():
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


t_total = time.time()
n_epochs_stop = 10
epochs_no_improve = 0
early_stop = False
min_val_loss = np.Inf

for epoch in range(args.epochs):
    val_loss = train(epoch)
    if early_stop:
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            # print("Early Stopping")
            early_stop = True
            break
        else:
            continue

print("Training completed!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()
