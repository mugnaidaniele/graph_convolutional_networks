import numpy as np
import scipy.sparse as sp
import torch


def encode_one_hot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_one_hot = np.array(list(map(classes_dict.get, labels)),
                              dtype=np.int32)
    return labels_one_hot


def load_data(path="cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = (idx_features_labels[:, 1:-1]).astype(np.float32)
    labels = encode_one_hot(idx_features_labels[:, -1])

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = np.eye(features.shape[0])
    for coord in edges:
        adj[coord[0], coord[1]] = 1
        adj[coord[1], coord[0]] = 1

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(matrix):
    row_sum = np.array(matrix.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    matrix = r_mat_inv.dot(matrix)
    return matrix


def accuracy(output, labels):
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
