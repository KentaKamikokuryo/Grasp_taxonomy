import math, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from Classes.Info import IDataInfo, DataInfo1, DataInfo2
from Classes.Data import DataFactory, Creation
from sklearn import decomposition
from Classes.Data import DataReader


class Spline():

    def __init__(self, indices: bytearray, data_info: IDataInfo, df_1: tuple = None, df_2: tuple = None, spline: int = 50):

        self.data_info = data_info

        # call the path for spline
        path_dict = self.data_info.get_data_info()
        self.result_path = path_dict["result"]
        self.data_path = path_dict["data"]

        # call the data merged
        self.value = self.merge_data(df_1=df_1, df_2=df_2)

        # call the cond for spline
        self.indices_segment = indices
        self.spline = spline

        self.output_data()

    def output_data(self):

        self.data_cut_X = []
        self.label_points = []
        self.labels = []
        n = 0

        self.catted_data = self.cut(indices_segment=self.indices_segment)

        for i, point in enumerate(self.catted_data):

            label_data = i
            self.label_points.append(label_data)
            self.labels.append(" Flame: " + str(point[0]) + " to " + str(point[1]))

            print(" Begin: " + str(point[0]) + " to " + str(point[1]))

            # Select the data for the current repetitions from begin to end (from point[0] to point[1])
            temp = self.value[int(point[0]): int(point[1])]

            # Spline the repetition to the same number of frame
            temp = self.spline_data(temp, spline=self.spline)
            self.data_cut_X.append(temp)

            n += 1

        self.database_X = np.concatenate([d[np.newaxis, :, :] for d in self.data_cut_X], axis=0)
        self.database_Y = np.array(self.label_points)
        self.labels = np.array(self.labels)
        self.database_X_std = np.empty(shape=self.database_X.shape)

        for i in range(self.database_X.shape[2]):
            self.mean = np.mean(self.database_X[:, :, i])
            self.std = np.std(self.database_X[:, :, i])
            self.database_X_std[:, :, i] = (self.database_X[:, :, i] - self.mean) / self.std

        self.save_data(DB_X=self.database_X, DB_Y=self.database_Y, Database_std=self.database_X_std, labels=self.labels, result_path=self.result_path)

    @staticmethod
    def merge_data(df_1, df_2):

        if df_1 is None:

            df_value = df_2[0]
            value = df_value.values

        elif df_2 is None:

            df_value = df_1[0]
            value = df_value.values

        else:

            df_1_value = df_1[0]
            df_2_value = df_2[0]

            df_value = pd.concat([df_1_value, df_2_value], axis=1)

            value = df_value.values

        return value

    @staticmethod
    def cut(indices_segment: bytearray):

        catted_data = np.zeros((len(indices_segment) - 1, 2))

        print("cutting data...")
        for i in tqdm(range(catted_data.shape[0])):

            d = np.zeros(2)
            d[0] = indices_segment[i]
            d[1] = indices_segment[i + 1] - 1
            catted_data[i] = d

        return catted_data

    @staticmethod
    def spline_data(data, spline=60):

        x = np.array([x for x in range(data.shape[0])])
        x_new = np.linspace(x.min(), x.max(), spline)
        data_spline = interp1d(x, data, kind='cubic', axis=0)(x_new)

        return data_spline

    @staticmethod
    def save_data(DB_X, DB_Y, Database_std, labels, result_path):

        np.save(result_path + "DB_X.npy", DB_X)
        print("Have saved Database_X")

        np.save(result_path + "DB_Y.npy", DB_Y)
        print("Have saved Database_Y")

        np.save(result_path + "DB_std.npy", Database_std)
        print("Have saved Data Standard deviation")

        np.save(result_path + "labels", labels)
        print("Have saved Data about labels")

class Activation_function():

    @staticmethod
    def sigmoid(x):
        x[x > 709.] = 709.
        return 1 / (1 + np.exp(-x))
