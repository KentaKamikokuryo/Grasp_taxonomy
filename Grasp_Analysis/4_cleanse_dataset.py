import os
import math
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classes.Data import DataFactory, Creation_Grasp
from Classes.Factories import DataInfoFactory
from Classes.Data import DataReader
from Classes.Info import IDataInfo
from Classes.Model_dimensionality_reduction import ModelDimensionalityReduction


class Manager():

    def __init__(self, experiment_name: str, take_names_list: List = None, running_info: Dict = None):

        self.experiment_name = experiment_name

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.take_names_list = take_names_list if take_names_list is not None else fac.get_take_names(name=self.experiment_name)

        self.retrieving_mode = running_info["retrieving_mode"]
        self.criteria_take = running_info["criteria_take"]

        self.on_PCA = running_info["on_PCA"]
        self.mode_PCA = running_info["mode_PCA"]
        self.compression_ratio_PCA = running_info["compression_ratio_PCA"]
        self.n_components_PCA = running_info["n_components_PCA"]

        self.on_manifold = running_info["on_manifold"]
        self.method_manifold = running_info["method_manifold"]
        self.n_components_manifold = running_info["n_components_manifold"]

    def _generate_average_data_dict(self):

        self.name_file = "grasp_cleansing_average"

        self.data_info.set_data_info(name_take=self.criteria_take)
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

    def _generate_all_data_dict(self):

        self.name_file = "grasp_cleansing_all"

        self.data_info.set_data_info(name_take=self.criteria_take)
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

    def _generate_dataset(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.dataset_df = data_factory.create()

        self.x = self.dataset_df[0].values
        self.y = self.dataset_df[3].values[:, 1]
        self.id = self.dataset_df[3].values[:, 0]

    def _apply_PCA(self):

        if self.mode_PCA == "components":
            self.hyper_dict_PCA = {"method_name": "PCA",
                                   "n_components": self.n_components_PCA}

        elif self.mode_PCA == "ratio":
            self.hyper_dict_PCA = {"method_name": "RPCA",
                                   "cumulative_contribution_ratio": self.compression_ratio_PCA}

        self.model_PCA = ModelDimensionalityReduction(hyper_dict=self.hyper_dict_PCA)

        self.x = self.model_PCA.fit(x_fit=self.x, y_fit=self.y)

    def _apply_manifold_learning(self):

        self.n_components_manifold = self.x.shape[1] if self.n_components_manifold is None else self.n_components_manifold

        if self.method_manifold == "LLE":
            self.hyper_dict_manifold = {"method_name": "LLE",
                                        "n_neighbors": 5,
                                        "n_components": self.n_components_manifold}

        elif self.method_manifold == "ISOMAP":
            self.hyper_dict_manifold = {"method_name": "ISOMAP",
                                        "n_neighbors": 5,
                                        "n_components": self.n_components_manifold}

        elif self.method_manifold == "TSNE":
            self.hyper_dict_manifold = {"method_name": "TSNE",
                                        "perplexity": 30.0,
                                        "learning_rate": 200.0,
                                        "n_components": self.n_components_manifold}

        self.model_manifold = ModelDimensionalityReduction(hyper_dict=self.hyper_dict_manifold)

        # FIXME: Modify so that it can deal with the case of t-SNE
        self.x = self.model_manifold.fit(x_fit=self.x, y_fit=self.y)

    def _concatenate_cleansed_data(self):

        all_values = np.concatenate([self.id, self.y, self.x], axis=1)

        header_label = ["Id", "Name"]
        header_values = ["Component_" + str(i+1) for i in range(self.x.shape[1])]
        header = header_label + header_values

        self.df_cleansed_dataset = pd.DataFrame(all_values, columns=header)

    def _save_dataset(self):

        self.path_dataset = self.data_dict["cleansing"]

        # REVIEW: Unknown if it will work
        self.df_cleansed_dataset.to_csv(self.path_dataset, header=True, index=False)

    def run(self):

        if self.retrieving_mode == "average":
            self._generate_average_data_dict()

        elif self.retrieving_mode == "all":
            self._generate_all_data_dict()

        self._generate_dataset()

        if self.on_PCA:
            self._apply_PCA()

        if self.on_manifold:
            self._apply_manifold_learning()

        self._concatenate_cleansed_data()
        self._save_dataset()


#------------------------------------------------------parameters------------------------------------------------------#
experiment_num = 2
take_names_list = ["WRK-T-A-15"]

running_info = {"retrieving_mode": "average",
                # "retrieving_mode": "all",
                "criteria_take": "WRK-T-A-15",

                "on_PCA": False,
                "mode_PCA": "ratio",
                "compression_ratio_PCA": 0.9,
                "n_components_PCA": 30,

                "on_manifold": False,
                "method_manifold": "ISOMAP",
                "n_components_manifold": None}
#----------------------------------------------------------------------------------------------------------------------#


experiment_names = ["experiment_1", "experiment_2", "experiment_3"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, take_names_list=take_names_list, running_info=running_info)
manager.run()
