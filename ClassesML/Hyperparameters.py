import numpy as np
import pandas as pd
import math, itertools, os
from tabulate import tabulate
from ClassesML.Models import Model

class Hyperparameters():

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hypermodel(self, display=False):

        grid = {}

        if self.model_name == Model.RF:

            grid = {"n_estimators": [10, 50, 100, 200, 300],
                    "min_samples_leaf": [1, 2, 3, 5, 10],
                    "criterion": ["gini"],
                    "class_weight": ["balanced"],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.SVC:

            grid = {"C": [0.01, 1, 10, 100],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "decision_function_shape": ["ovo", "ovr"],
                    "class_weight": ["balanced"],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.KNN:

            grid = {"n_neighbors": [3, 5, 10],
                    "n_jobs": [1, 2],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.LR:

            grid = {"C": [0.01, 1, 2, 3, 4],
                    "random_state": [10, 40, 120, 200],
                    "max_iter": [3000],
                    "class_weight": ["balanced"],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.AB:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.GB:

            grid = {"model_name": [self.model_name]}

        elif self.model_name == Model.CB:

            grid = {"iterations": [10, 20, 30],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.XGB:

            grid = {"n_estimators": [10, 50, 100, 1000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5],
                    "max_depth": [3, 6, 10, 20],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.LGBM:

            grid = {"n_estimators": [10, 50, 100, 1000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5],
                    "max_depth": [3, 6, 10, 20],
                    "class_weight": ["balanced"],
                    "model_name": [self.model_name]}

        elif self.model_name == Model.NB:

            grid = {"var_smoothing": [0.00000001, 0.000000001, 0.0000000001],
                    "model_name": [self.model_name]}

        else:

            grid = dict()

        keys, values = zip(*grid.items())
        grid_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(grid_combination)

        if display:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return grid_combination, grid

