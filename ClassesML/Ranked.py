import numpy as np
import pandas as pd
from tabulate import tabulate

class Ranked():

    mean_list_sorted: list
    std_list_sorted: list

    def __init__(self, model_name, path: str="",
                 metric: str = "f1",
                 sort_type: str="ascending",
                 metric_type: str="accuracy"):

        self.model_name = model_name
        self.path = path

        self.sort_type = sort_type
        self.metric_type = metric_type

        self.hyperparameters_list = []
        self.mean_list = []
        self.std_list = []

        self.name = "Model_" + str(self.model_name) + "_Metric_" + str(metric)

    def add(self, hyperparameter, mean=0, std=0):

        self.hyperparameters_list.append(hyperparameter)
        self.mean_list.append(mean)
        self.std_list.append(std)

    def ranked(self, display: bool = True, save: bool = True):

        if self.metric_type == "ascending":
            idx = np.argsort(self.mean_list)[::-1]
        elif self.metric_type == "descending":
            idx = np.argsort(self.mean_list)
        else:
            idx = np.argsort(self.mean_list)[::-1]

        self._hyperparameters_list_sorted = np.array(self.hyperparameters_list)[idx].tolist()
        self.mean_list_sorted = np.array(self.mean_list)[idx].tolist()
        self.std_list_sorted = np.array(self.std_list)[idx].tolist()

        print("Hyperparameters_list has been ranked")

        self._hyperparameter_best = self._hyperparameters_list_sorted[0]

        df = pd.DataFrame.from_dict(self._hyperparameters_list_sorted)
        df["mean"] = self.mean_list_sorted
        df["std"] = self.std_list_sorted

        if display:

            print(tabulate(df, headers="keys", tablefmt="psql"))

        if save:

            name = "ranked_" + self.name + ".csv"
            df.to_csv(self.path + name)
            print("Hyperparameters_list_sorted has been saved to: " + self.path + name)

    def save_best_hyperparameter(self):

        name = self.name + "_best.npy"

        np.save(self.path + name, self._hyperparameter_best)
        print("Hyperparameter_best has been saved to: " + self.path + name)

    def load_best_hyperparameter(self):

        name = self.name + "_best.npy"

        self._hyperparameter_best = np.load(self.path + name, allow_pickle=True).item()
        print("Hyperparameter_best has been loaded from: " + self.path + name)

    def load_ranked_list(self, display: bool = False):

        name = "ranked_" + self.name + ".csv"

        self._hyperparameters_list_sorted = pd.read_csv(self.path + name)
        print("Hyperparameters_ranked has been load from: " + self.path + name)

        if display:

            df = pd.DataFrame.from_dict(self._hyperparameters_list_sorted)
            print(tabulate(df, headers="keys", tablefmt="psql"))

    @property
    def hyperparameter_best(self):
        return self._hyperparameter_best

    @property
    def hyperparameters_list_sorted(self):
        return self._hyperparameters_list_sorted




