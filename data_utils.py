import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_content(content, cities):
    le_index = LabelEncoder()
    le_label = LabelEncoder()

    df = pd.read_csv(content, sep="\t", header=None)
    df_cities = pd.read_csv(cities, sep="\t", header=None)

    df[0] = le_index.fit_transform(df[0].values)
    df[df.shape[1] - 1] = le_label.fit_transform(df[df.shape[1] - 1].values)
    label = df[df.shape[1] - 1]
    df = df.drop(df.shape[1] - 1, axis=1)
    df = df.drop(0, axis=1)

    num_lines = sum(1 for line in open(content))
    adj = np.zeros((num_lines, num_lines)) + np.eye(num_lines)

    df_cities[0] = le_index.transform(df_cities[0].values)
    df_cities[df_cities.shape[1] - 1] = le_index.transform(df_cities[df_cities.shape[1] - 1].values)

    cit = df_cities.to_numpy()
    for coords in cit:
        x, y = coords[0], coords[1]
        adj[x][y] = 1
        adj[y][x] = 1

    label = label.to_numpy()
    features = df.to_numpy()

    return adj, features, label


# TODO SPLIT DATASET
# TODO NORMALIZE
adj, features, label = get_content("cora/cora.content", "cora/cora.cites")
