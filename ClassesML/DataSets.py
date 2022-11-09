import numpy as np
import pandas as pd
import os
from sklearn import datasets
from ClassesML.Interfaces import IDataSet
from tabulate import tabulate
from Classes.Factories import DataInfoFactory
from ClassesML.Features import Kinematics_both_hands, Kinematics_one_hand
from ClassesML.InfoML import PathInfoML

class DataSets:

    Grasp_both = "grasp_both_hands"
    Grasp_one = "grasp_one_hand"
    Iris = "iris"
    Wine = "wine"
    Digits = "digits"

class DataInfo:

    def __init__(self):

        self.data_names = [DataSets.Grasp_both,
                           DataSets.Grasp_one,
                           DataSets.Iris,
                           DataSets.Wine,
                           DataSets.Digits]

        self.data_names = [DataSets.Grasp_one]

        self.data_info = {"test_size": 0.35,
                          "n_K_fold": 5,
                          "k_shuffle": True,
                          "data_mode": "all",
                          "hand_mode": "one"}

    def get_dataset_names(self):

        return self.data_names

    def get_data_info(self):

        return self.data_info

class GraspBothHands(IDataSet):

    def __init__(self, df_input, display: bool=False):

        self._df_input = df_input
        self.display = display

    def create(self):

        kinematics = Kinematics_both_hands(df_input=self._df_input, drop_corr_features=True)

        df = kinematics.df
        X = kinematics.X
        y = kinematics.y
        feature_names = kinematics.feature_names
        target_names = kinematics.target_names

        if self.display:

            print(df.head())

        return X, y, feature_names, target_names, df

    @property
    def df_input(self):
        return self._df_input

class GraspOneHand(IDataSet):

    def __init__(self, df_input, display: bool=False):

        self._df_input = df_input
        self.display = display

    def create(self):

        kinematics = Kinematics_one_hand(df_input=self._df_input, drop_corr_features=True)

        df = kinematics.df
        X = kinematics.X
        y = kinematics.y
        feature_names = kinematics.feature_names
        target_names = kinematics.target_names

        if self.display:

            print(df.head())

        return X, y, feature_names, target_names, df

    @property
    def df_input(self):
        return self._df_input

class Iris(IDataSet):

    def __init__(self, display: bool=False):

        self.display = display

    def create(self):

        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        index = iris.feature_names
        class_names = iris.target_names

        if self.display:

            print(df.head())

        X, y = datasets.load_iris(return_X_y=True)

        return X, y, index, class_names, df


class Wine(IDataSet):

    def __init__(self, display: bool = False):
        self.display = display

    def create(self):

        wine = datasets.load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        index = wine.feature_names
        class_names = wine.target_names

        if self.display:
            print(df.head())

        X, y = datasets.load_wine(return_X_y=True)

        return X, y, index, class_names, df


class Digits(IDataSet):

    def __init__(self, display: bool = False):
        self.display = display

    def create(self):

        digits = datasets.load_digits()
        df = pd.DataFrame(digits.data, columns=digits.feature_names)
        index = digits.feature_names
        class_names = digits.target_names

        if self.display:
            print(df.head())

        X, y = datasets.load_digits(return_X_y=True)

        return X, y, index, class_names, df