import math, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn import decomposition
from Classes.Info import PathInfo, DataInfo1
from Classes.Data import CSVDataStrategy, NumpySegmentDBStrategy, NumpyAllIndicesStrategy
from Classes.Data import DataFactory, Creation

name = "take-A-3"
name_file_DB = "segment_DB"
dataInfo_v = DataInfo1(name_take=name, name_file=name_file_DB)
data_manager_v = DataFactory(data_info=dataInfo_v)
DB_X, DB_Y, DB_std, labels = data_manager_v.create()
labels = list(labels)

def get_data(DB_X, DB_Y, test: bool= False ,shuffle: bool= False):

    p = np.array([i for i in range(DB_X.shape[0])])
    if shuffle:
        random.shuffle(p)

    n = DB_X.shape[0]

    if test:
        n_sample_train = int(n * 0.8)
    else:
        n_sample_train = int(n * 1)

    train_X = DB_X[p][:n_sample_train]
    train_X = np.transpose(train_X, [0, 2, 1])
    train_X = train_X.reshape([train_X.shape[0], -1])

    train_Y = DB_Y[p][:n_sample_train]

    if test:
        test_X = DB_X[p][n_sample_train:]
        test_X = np.transpose(test_X, [0, 2, 1])
        test_X = test_X.reshape([test_X.shape[0], -1])

        test_Y = DB_Y[p][n_sample_train:]
    else:
        pass

    return train_X, train_Y

train_X, train_Y= get_data(DB_X, DB_Y)

model = decomposition.PCA(n_components=2)
model.fit(train_X)

print(model.components_)

z_train = model.transform(train_X)  # Transform

unique_class = np.unique(train_Y)

cmap = plt.get_cmap("nipy_spectral")
colors = cmap(np.linspace(0,1, len(unique_class)))
colors = dict(zip(unique_class, colors))

plt.figure()

for ind in unique_class:

    indices = [train_Y == ind]

    X = z_train[:, 0][indices]
    Y = z_train[:, 1][indices]

    plt.scatter(X, Y, color=colors[ind], label=labels[ind], s=300)

    # indices = [test_Y == ind]
    #
    # X = z_test[:, 0][indices]
    # Y = z_test[:, 1][indices]
    #
    # plt.scatter(X, Y, color=colors[ind], label=labels[ind], s=300, marker="+")

    plt.legend()