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


class Manager():

    def __init__(self, experiment_name: str, take_names_list: List = None, running_info: Dict = None):

        self.experiment_name = experiment_name

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.take_names_list = take_names_list if take_names_list is not None else fac.get_take_names(name=self.experiment_name)

        self.retrieving_mode = running_info["retrieving_mode"]
        self.hand_mode = running_info["hand_mode"]
        self.criteria_take = running_info["criteria_take"]

    def _integrate_takes_data(self):

        self.takes_values_ndarrays = []
        self.takes_identifiers_ndarrays = []

        for take in self.take_names_list:

            self.take_name = take

            self._generate_data_dict()

            self._generate_take_data()

            self.takes_values_ndarrays.append(self.take_df[0].values)
            self.takes_identifiers_ndarrays.append(self.take_df[2].values)

            if self.take_name == self.criteria_take:
                self.criteria_properties = self.take_df[1]

        self.takes_values = np.concatenate(self.takes_values_ndarrays, axis=0)
        self.takes_identifiers = np.concatenate(self.takes_identifiers_ndarrays, axis=0)

    def _generate_data_dict(self):

        self.name_file = "grasp_labeling"

        self.data_info.set_data_info(name_take=self.take_name)
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

        if self.retrieving_mode in ["average", "all", "point"]:
            self.data_dict["name"] = self.data_dict["posture_name"]
        elif self.retrieving_mode in ["duration"]:
            self.data_dict["name"] = self.data_dict["motion_name"]

        if self.retrieving_mode == "average":
            self.data_dict["data"] = self.data_dict["average_data"]
            self.data_dict["dataset"] = self.data_dict["average_dataset"]
        elif self.retrieving_mode == "all":
            self.data_dict["data"] = self.data_dict["all_data"]
            self.data_dict["dataset"] = self.data_dict["all_dataset"]
        elif self.retrieving_mode == "point":
            self.data_dict["data"] = self.data_dict["point_data"]
            self.data_dict["dataset"] = self.data_dict["point_dataset"]
        elif self.retrieving_mode == "duration":
            self.data_dict["data"] = self.data_dict["duration_data"]
            self.data_dict["dataset"] = self.data_dict["duration_dataset"]

        if self.hand_mode == "both":
            self.data_dict["data"] = self.data_dict["data"].split(".")[0] + "_both.csv"
            self.data_dict["dataset"] = self.data_dict["dataset"].split(".")[0] + "_both.csv"
        elif self.hand_mode == "one":
            self.data_dict["data"] = self.data_dict["data"].split(".")[0] + "_one.csv"
            self.data_dict["dataset"] = self.data_dict["dataset"].split(".")[0] + "_one.csv"

    def _generate_take_data(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.take_df = data_factory.create()

    def _revise_properties_values(self):

        self.criteria_properties_values = self.criteria_properties.values

        if self.retrieving_mode in ["average", "all", "point"]:
            self.part_names = self.criteria_properties_values[1, 2:]
            self.pos_rot_names = self.criteria_properties_values[3, 2:]
            self.axis_names = self.criteria_properties_values[4, 2:]
            self.identifier_names = self.criteria_properties_values[-1, :2]

        elif self.retrieving_mode in ["duration"]:
            self.part_names = self.criteria_properties_values[1, 3:]
            self.pos_rot_names = self.criteria_properties_values[3, 3:]
            self.axis_names = self.criteria_properties_values[4, 3:]
            self.identifier_names = self.criteria_properties_values[-1, :3]

        self.revised_property_names = np.empty([self.part_names.shape[0]], dtype=object)

        for i in range(self.part_names.shape[0]):

            self.revised_property_names[i] = self.part_names[i]
            # self.revised_property_names[i] +=\
            #     "_pos" if self.pos_rot_names[i] == "Position" else "_rot" if self.pos_rot_names[i] == "Rotation" else "_non"
            self.revised_property_names[i] += \
                "_Position" if self.pos_rot_names[i] == "Position" else "_Rotation" if self.pos_rot_names[i] == "Rotation" else "_Non"
            self.revised_property_names[i] += "_" + self.axis_names[i]

        self.revised_property_names = np.concatenate([self.identifier_names, self.revised_property_names])

    def _concatenate_takes_data(self):

        identifiers_values = self.takes_identifiers
        values_values = self.takes_values
        properties_values = self.revised_property_names

        all_values = np.concatenate([identifiers_values, values_values[:, 2:]], axis=1) if self.retrieving_mode in ["average", "all", "point"]\
            else np.concatenate([identifiers_values, values_values[:, 3:]], axis=1) if self.retrieving_mode in ["duration"]\
            else None
        dataset_values = np.concatenate([properties_values.reshape([1, -1]), all_values], axis=0)

        self.df_dataset = pd.DataFrame(dataset_values)

    def _save_dataset(self):

        self.path_dataset = self.data_dict["dataset"]

        self.df_dataset.to_csv(self.path_dataset, header=False, index=False)

    def run(self):

        self._integrate_takes_data()
        self._revise_properties_values()
        self._concatenate_takes_data()
        self._save_dataset()


#------------------------------------------------------parameters------------------------------------------------------#
# experiment_num = 2
experiment_num = 5
# take_names_list = ["WRK-M-A-5", "WRK-M-A-6", "WRK-M-A-8", "WRK-M-A-9", "WRK-M-A-11"]
# take_names_list = ["tera_0_202206241644", "tera_0_202206241653", "tera_0_202206241655",
#                    "tera_1_202206241658", "tera_1_202206241703", "tera_1_202206241705", "tera_1_202206241707",
#                    "tera_2_202206241708", "tera_2_202206241710", "tera_2_202206241711",
#                    "tera_3_202206241724", "tera_3_202206241727", "tera_3_202206241729",
#                    "tera_4_202206241732", "tera_4_202206241736", "tera_4_202206241738"]
take_names_list = ["haga_0_202208251114", "haga_0_202208251126", "haga_0_202208251131",
                   "haga_1_202208251143", "haga_1_202208251149", "haga_1_202208251153",
                   "haga_2_202208251158", "haga_2_202208251200", "haga_2_202208251202",
                   "haga_3_202208251221", "haga_3_202208251225", "haga_3_202208251228",
                   "haga_4_202208251232", "haga_4_202208251238", "haga_4_202208251241",
                   "haya_0_202208251608", "haya_0_202208251617", "haya_0_202208251625",
                   "haya_1_202208251646", "haya_1_202208251651", "haya_1_202208251656",
                   "haya_2_202208251700", "haya_2_202208251704", "haya_2_202208251707",
                   "haya_3_202208251743", "haya_3_202208251746", "haya_3_202208251748",
                   "haya_4_202208251752", "haya_4_202208251759", "haya_4_202208251803",
                   "kami_0_202208251442", "kami_0_202208251446", "kami_0_202208251451",
                   "kami_1_202208251435", "kami_1_202208251439", "kami_1_202208251441",
                   "kami_2_202208251430", "kami_2_202208251432", "kami_2_202208251433",
                   "kami_3_202208251409", "kami_3_202208251412", "kami_3_202208251413",
                   "kami_4_202208251357", "kami_4_202208251402", "kami_4_202208251405",
                   "tera_0_202206241644", "tera_0_202206241653", "tera_0_202206241655",
                   "tera_1_202206241658", "tera_1_202206241703", "tera_1_202206241705", "tera_1_202206241707",
                   "tera_2_202206241708", "tera_2_202206241710", "tera_2_202206241711",
                   "tera_3_202206241724", "tera_3_202206241727", "tera_3_202206241729",
                   "tera_4_202206241732", "tera_4_202206241736", "tera_4_202206241738"]

running_info = {# "retrieving_mode": "average",
                # "retrieving_mode": "all",
                # "retrieving_mode": "point",
                "retrieving_mode": "duration",
                # "hand_mode": "both",
                "hand_mode": "one",
                # "criteria_take": "WRK-M-A-5"
                "criteria_take": "tera_0_202206241644"}
#----------------------------------------------------------------------------------------------------------------------#


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, take_names_list=take_names_list, running_info=running_info)
manager.run()
