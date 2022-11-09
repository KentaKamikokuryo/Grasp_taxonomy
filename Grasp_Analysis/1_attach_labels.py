import os
import math
from typing import List, Dict
import numpy as np
import pandas as pd
import rapidjson
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tabulate import tabulate
from Classes.Data import DataFactory, Creation_Grasp
from Classes.Factories import DataInfoFactory
from Classes.Data import DataReader
from Classes.Info import IDataInfo
from Classes.Console_utilities import Color


class Manager():

    def __init__(self, experiment_name: str, take_name: str, running_info: Dict):

        self.experiment_name = experiment_name
        self.take_name = take_name

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.take_names = fac.get_take_names(name=self.experiment_name)

        self.running_mode = running_info["running_mode"]
        self.retrieving_mode = running_info["retrieving_mode"]
        self.hand_mode = running_info["hand_mode"]

        self.work_key = running_info["work_key"]
        self.bend_key = running_info["bend_key"]
        self.both_label_key = running_info["both_label_key"]
        self.left_label_key = running_info["left_label_key"]
        self.right_label_key = running_info["right_label_key"]

        self.work_value = running_info["work_value"]
        self.both_bend_values = running_info["both_bend_values"]
        self.left_bend_values = running_info["left_bend_values"]
        self.right_bend_values = running_info["right_bend_values"]
        self.bend_names = running_info["bend_names"]

    # for "auto" running_mode
    def _initialize_auto_mode(self):

        self.experiment_dict = self.data_info.get_experiment_info()

        self.takes_json = rapidjson.load(open(self.experiment_dict["take_info"], 'r'))
        self.take_name_list = self.takes_json["take_name_list"]
        self.work_num_list = self.takes_json["work_num_list"]
        self.count_num_list = self.takes_json["count_num_list"]
        self.bend_nums_list = self.takes_json["bend_nums_list"]
        self.bend_names_list = self.takes_json["bend_names_list"]

    # for "auto" running_mode
    def _initialize_running_info_for_take(self):

        self.take_name = self.take_name_list[self.take_i]

        self.work_value = self.work_num_list[self.take_i]
        self.both_bend_values = self.bend_nums_list[self.take_i]
        self.left_bend_values = self.bend_nums_list[self.take_i]
        self.right_bend_values = self.bend_nums_list[self.take_i]
        self.bend_names = self.bend_names_list[self.take_i]

    def _generate_data_dict(self):

        self.name_file = "grasp_retrieving"

        self.data_info.set_data_info(name_take=self.take_name)
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

        if self.retrieving_mode in ["average", "all", "point"]:
            self.data_dict["name"] = self.data_dict["posture_name"]
        elif self.retrieving_mode in ["duration"]:
            self.data_dict["name"] = self.data_dict["motion_name"]

        if self.retrieving_mode == "average":
            self.data_dict["data"] = self.data_dict["average_data"]
            self.data_dict["result"] = self.data_dict["average_result"]
        elif self.retrieving_mode == "all":
            self.data_dict["data"] = self.data_dict["all_data"]
            self.data_dict["result"] = self.data_dict["all_result"]
        elif self.retrieving_mode == "point":
            self.data_dict["data"] = self.data_dict["point_data"]
            self.data_dict["result"] = self.data_dict["point_result"]
        elif self.retrieving_mode == "duration":
            self.data_dict["data"] = self.data_dict["duration_data"]
            self.data_dict["result"] = self.data_dict["duration_result"]

        if self.hand_mode == "both":
            self.data_dict["result"] = self.data_dict["result"].split(".")[0] + "_both.csv"
        elif self.hand_mode == "one":
            self.data_dict["result"] = self.data_dict["result"].split(".")[0] + "_one.csv"

    def _generate_original_data(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.original_df = data_factory.create()

        self.original_df_value = self.original_df[0]
        self.original_df_properties = self.original_df[1]
        self.original_df_header = self.original_df[2]
        self.original_df_identifier = self.original_df[3]

        if self.hand_mode == "one":

            self.unique_ids = list(np.unique(self.original_df_identifier["Id"].values))
            self.unique_ids_indices = [self.original_df_identifier.query('Id == "{0}"'.format(unique_id)).index
                                       for unique_id in self.unique_ids]
            self.motion_frame_num = len(self.unique_ids_indices[0])

            original_df_value_values = self.original_df_value.values
            original_df_value_values = np.array([original_df_value_values[self.unique_ids_indices[unique_id_index]]
                                                 for unique_id_index in range(len(self.unique_ids))
                                                 for L_R in ["L", "R"]])\
                .reshape([-1, original_df_value_values.shape[1]])
            # original_df_value_values = np.array([original_df_value_values[int(row/2)]
            #                                      for row in range(len(original_df_value_values)*2)])

            original_df_identifier_values = self.original_df_identifier.values
            original_df_identifier_values = np.array([original_df_identifier_values[self.unique_ids_indices[unique_id_index]]
                                                      for unique_id_index in range(len(self.unique_ids))
                                                      for L_R in ["L", "R"]])\
                .reshape([-1, original_df_identifier_values.shape[1]])
            # original_df_identifier_values = np.array([original_df_identifier_values[int(row/2)]
            #                                           for row in range(len(original_df_identifier_values)*2)])

            original_df_values_L_indices = list(np.array([list(np.array(self.unique_ids_indices[unique_id_index])
                                                               + self.motion_frame_num * unique_id_index)
                                                          for unique_id_index in range(len(self.unique_ids))]).flatten())
            original_df_values_R_indices = list(np.array([list(np.array(self.unique_ids_indices[unique_id_index])
                                                               + self.motion_frame_num * unique_id_index
                                                               + self.motion_frame_num)
                                                          for unique_id_index in
                                                          range(len(self.unique_ids))]).flatten())

            original_df_identifier_values[original_df_values_L_indices, 0:2] += "_L"
            original_df_identifier_values[original_df_values_R_indices, 0:2] += "_R"
            # for row in range(len(original_df_identifier_values)):
            #     original_df_identifier_values[row, 0] += "_L" if row % 2 == 0 else "_R"
            #     original_df_identifier_values[row, 1] += "_L" if row % 2 == 0 else "_R"

            self.original_df_value = pd.DataFrame(data=original_df_value_values, columns=self.original_df_value.columns)
            self.original_df_identifier = pd.DataFrame(data=original_df_identifier_values, columns=self.original_df_identifier.columns)

    def _load_bendinfo_csv(self):

        self.bendinfo_df = pd.read_csv(self.data_dict["bend_info_csv"], sep=",")

    def _overwrite_identifier(self):

        self.work_bendinfo_df = self.bendinfo_df[self.bendinfo_df[self.work_key] == self.work_value]

        if self.hand_mode == "both":

            self.bend_values = self.both_bend_values

            self.work_label_series = self.work_bendinfo_df[self.both_label_key]

        elif self.hand_mode == "one":

            self.bend_values = self.left_bend_values + self.right_bend_values

            self.bend_names = [self.bend_names[i] + "_L" for i in range(len(self.bend_names))] \
                              + [self.bend_names[i] + "_R" for i in range(len(self.bend_names))]

            self.work_label_series = pd.concat([self.work_bendinfo_df[self.left_label_key], self.work_bendinfo_df[self.right_label_key]])

        self.bend_labels = []

        for i, bend_value in enumerate(self.bend_values):

            work_label_series_indices = self.work_bendinfo_df.reset_index().query('{0} == {1}'.format(self.bend_key, bend_value)).index
            work_label_series_index = work_label_series_indices[0] if i < len(self.work_bendinfo_df)\
                else work_label_series_indices[0]+len(self.work_bendinfo_df)

            self.bend_labels.append(self.work_label_series.iat[work_label_series_index])

        self.bend_name_label_zip = zip(self.bend_names, self.bend_labels)

        for name, label in self.bend_name_label_zip:
            self.original_df_identifier.replace({"Name": {name: label}}, inplace=True)

        self.original_df_properties.iat[-1, 1] = "Label"

    def _concatenate_overwritten_dataframes(self):

        overwritten_value_values = self.original_df_value.values[:, 2:] if self.retrieving_mode in ["average", "all", "point"]\
            else self.original_df_value.values[:, 3:] if self.retrieving_mode in ["duration"]\
            else None
        overwritten_properties_values = self.original_df_properties.values
        overwritten_header_values = self.original_df_header.values
        overwritten_identifier_values = self.original_df_identifier.values

        below_values = np.concatenate([overwritten_identifier_values, overwritten_value_values], axis=1)
        except_header_values = np.concatenate([overwritten_properties_values, below_values], axis=0)

        difference_header_column = except_header_values.shape[1] - len(self.original_df_header.columns)
        wider_values = "properties" if difference_header_column > 0 else "header" if difference_header_column < 0 else "equal"
        difference_header_column = abs(difference_header_column)

        if wider_values == "properties":

            nan_ndarray = np.empty([2, difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            overwritten_header_values = np.concatenate([overwritten_header_values, nan_ndarray], axis=1)

        elif wider_values == "header":

            nan_ndarray = np.empty([except_header_values.shape[0], difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            except_header_values = np.concatenate([except_header_values, nan_ndarray], axis=1)

        overwritten_values = np.concatenate([overwritten_header_values, except_header_values], axis=0)

        self.labeling_df = pd.DataFrame(overwritten_values)

    def _save_labeling_data(self):

        self.path_labeling_data = self.data_dict["result"]

        self.labeling_df.to_csv(self.path_labeling_data, header=False, index=False)

    def _run_manual_mode(self):

        self._generate_data_dict()
        self._generate_original_data()
        self._load_bendinfo_csv()
        self._overwrite_identifier()
        self._concatenate_overwritten_dataframes()
        self._save_labeling_data()

    def _run_auto_mode(self):

        self._initialize_auto_mode()

        for take_i in range(len(self.take_name_list)):

            self.take_i = take_i

            self._initialize_running_info_for_take()

            self._generate_data_dict()
            self._generate_original_data()
            self._load_bendinfo_csv()
            self._overwrite_identifier()
            self._concatenate_overwritten_dataframes()
            self._save_labeling_data()

    def run(self):

        if self.running_mode == "manual":
            self._run_manual_mode()

        elif self.running_mode == "auto":
            self._run_auto_mode()


# -----------------------------------------------------parameters----------------------------------------------------- #
# experiment_num = 2
experiment_num = 5
# take_name = "WRK-M-A-11"
take_name = "tera_2_202206241711"

running_info = {# "running_mode": "manual",
                "running_mode": "auto",
                # "retrieving_mode": "average",
                # "retrieving_mode": "all",
                # "retrieving_mode": "point",
                "retrieving_mode": "duration",
                # "hand_mode": "both",
                "hand_mode": "one",
                "work_key": "work_id",
                "bend_key": "process",
                "both_label_key": "g_ng",  # this variable is only used when hand_mode is "both"
                "left_label_key": "left_hand_posture",  # this variable is only used when hand_mode is "one"
                "right_label_key": "right_hand_posture",  # this variable is only used when hand_mode is "one"
                "work_value": 1,
                # "both_bend_values": [0, 1, 2, 3, 4, 5, 6, 7],
                # "left_bend_values": [0, 1, 2, 3, 4, 5, 6, 7],
                # "right_bend_values": [0, 1, 2, 3, 4, 5, 6, 7],
                "both_bend_values": [0, 1, 2, 3, 4],
                "left_bend_values": [0, 1, 2, 3, 4],
                "right_bend_values": [0, 1, 2, 3, 4],
                # "bend_names": ["bend_0", "bend_1", "bend_2", "bend_3", "bend_4", "bend_5", "bend_6", "bend_7"]
                "bend_names": ["bend_0", "bend_1", "bend_2", "bend_3", "bend_4"],
                }
# -------------------------------------------------------------------------------------------------------------------- #


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, take_name=take_name, running_info=running_info)
manager.run()
