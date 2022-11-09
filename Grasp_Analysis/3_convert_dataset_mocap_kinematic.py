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

    def __init__(self, experiment_name: str, running_info: Dict = None):

        self.experiment_name = experiment_name

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.take_names = fac.get_take_names(name=self.experiment_name)

        self.retrieving_mode = running_info["retrieving_mode"]
        self.hand_mode = running_info["hand_mode"]
        self.correspondence_mocap_kinematic_variable = running_info["correspondence_mocap_kinematic_variable"]
        self.identifier_columns = running_info["identifier_columns"]

        self.direct_pairs = self.correspondence_mocap_kinematic_variable["direct_pairs"]
        self.functions = self.correspondence_mocap_kinematic_variable["functions"]

    def _generate_data_dict(self):

        if self.retrieving_mode == "average":
            self.name_file = "grasp_dataset_mocap_average"
        elif self.retrieving_mode == "all":
            self.name_file = "grasp_dataset_mocap_all"
        elif self.retrieving_mode == "duration":
            self.name_file = "grasp_dataset_mocap_duration"

        self.data_info.set_data_info(name_take=self.take_names[0])
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

        if self.hand_mode == "both":
            self.data_dict["data"] = self.data_dict["data"].split(".")[0] + "_both.csv"
            self.data_dict["kinematic"] = self.data_dict["kinematic"].split(".")[0] + "_both.csv"
        elif self.hand_mode == "one":
            self.data_dict["data"] = self.data_dict["data"].split(".")[0] + "_one.csv"
            self.data_dict["kinematic"] = self.data_dict["kinematic"].split(".")[0] + "_one.csv"

    def _generate_mocap_data(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.mocap_df = data_factory.create()

    def _create_kinematic_variable_df(self):

        self._extract_direct_mocap_df()
        self._create_direct_kinematic_variable_df()

        self._extract_functions_mocap_dfs()
        self._compute_functions_mocap_dfs()
        self._create_functions_kinematic_variable_df()

        self._extract_identifiers_df()

        self.kinematic_variable_df\
            = pd.concat([self.identifiers_df, self.direct_kinematic_variable_df, self.functions_kinematic_variable_df],
                        axis=1)

    def _extract_direct_mocap_df(self):

        self.direct_mocap_re = ""

        for (mocap, kinematic_variable) in self.direct_pairs:
            self.direct_mocap_re += mocap + "|"

        self.direct_mocap_re = self.direct_mocap_re[:-1]
        self.direct_mocap_re = "(" + self.direct_mocap_re + ")"

        self.direct_mocap_df = self.mocap_df.filter(regex=self.direct_mocap_re)

    def _create_direct_kinematic_variable_df(self):

        self.direct_kinematic_variable_values = self.direct_mocap_df.values
        self.direct_kinematic_variable_columns = self.direct_mocap_df.columns.values.tolist()

        self.direct_kinematic_variable_columns = [column.replace(mocap, kinematic_variable)
                                                  for column in self.direct_kinematic_variable_columns
                                                  for (mocap, kinematic_variable) in self.direct_pairs
                                                  if mocap in column]

        self.direct_kinematic_variable_df = pd.DataFrame(data=self.direct_kinematic_variable_values,
                                                         columns=self.direct_kinematic_variable_columns)

    def _extract_functions_mocap_dfs(self):

        self.functions_mocap_re_list = []

        for (independents, kinematic_variable, function) in self.functions:

            functions_mocap_re = ""

            for independent in independents:

                functions_mocap_re += independent + "|"

            functions_mocap_re = functions_mocap_re[:-1]
            functions_mocap_re = "(" + functions_mocap_re + ")"

            self.functions_mocap_re_list.append(functions_mocap_re)

        self.functions_mocap_df_list = [self.mocap_df.filter(regex=re) for re in self.functions_mocap_re_list]

    def _compute_functions_mocap_dfs(self):

        self.functions_kinematic_variable_values_list = []

        for i, (_, _, function) in enumerate(self.functions):

            target_values = self.functions_mocap_df_list[i].values

            if function == "palm":

                self.functions_kinematic_variable_values_list.append(MocapKinematicVariableFunctions.palm(target_values))

    def _create_functions_kinematic_variable_df(self):

        self.functions_kinematic_variable_values = np.concatenate(self.functions_kinematic_variable_values_list, axis=1)

        self.functions_kinematic_variable_columns = [df.columns.values.tolist()[:3] for df in self.functions_mocap_df_list]
        self.functions_kinematic_variable_columns = [column
                                                     for row in self.functions_kinematic_variable_columns
                                                     for column in row]

        self.functions_kinematic_variable_columns = [column.replace(independents[0], kinematic_variable)
                                                     for (independents, kinematic_variable, _) in self.functions
                                                     for column in self.functions_kinematic_variable_columns
                                                     if independents[0] in column]

        self.functions_kinematic_variable_df = pd.DataFrame(data=self.functions_kinematic_variable_values,
                                                            columns=self.functions_kinematic_variable_columns)

    def _extract_identifiers_df(self):

        self.identifiers_df = self.mocap_df.filter(items=self.identifier_columns)

    def _convert_one_hand_df(self):

        if self.retrieving_mode in ["average", "all"]:
            self.one_hand_left_df = self.kinematic_variable_df.query('Id.str.endswith("_L")').filter(regex='(Id|Label|_L)', axis=1)
            self.one_hand_right_df = self.kinematic_variable_df.query('Id.str.endswith("_R")').filter(regex='(Id|Label|_R)', axis=1)

        elif self.retrieving_mode in ["duration"]:
            self.one_hand_left_df = self.kinematic_variable_df.query('Id.str.endswith("_L")').filter(regex='(Id|Label|Frame|_L)', axis=1)
            self.one_hand_right_df = self.kinematic_variable_df.query('Id.str.endswith("_R")').filter(regex='(Id|Label|Frame|_R)', axis=1)

        self.one_hand_left_df.rename(columns=lambda s: s.replace("_L", ""), inplace=True)
        self.one_hand_right_df.rename(columns=lambda s: s.replace("_R", ""), inplace=True)

        self.kinematic_variable_df = pd.concat([self.one_hand_left_df, self.one_hand_right_df], axis=0).sort_index()

    def _save_kinematic_variable_df(self):

        self.path_data = self.data_dict["kinematic"]

        self.kinematic_variable_df.to_csv(self.path_data, header=True, index=False)

    def run(self):

        self._generate_data_dict()
        self._generate_mocap_data()
        self._create_kinematic_variable_df()

        if self.hand_mode == "one":
            self._convert_one_hand_df()

        self._save_kinematic_variable_df()


class MocapKinematicVariableFunctions():

    @staticmethod
    def palm(x: np.ndarray):

        x1 = x[:, 0:3]
        x2 = x[:, 3:6]

        fx = np.mean(np.stack([x1, x2]), axis=0)

        return fx


#------------------------------------------------------parameters------------------------------------------------------#
# experiment_num = 2
experiment_num = 5

correspondence_mocap_kinematic_variable = {"direct_pairs": [("RightHandThumb4", "thumb_EF_R"),
                                                            ("RightHandThumb3", "thumb_D_R"),
                                                            ("RightHandThumb2", "thumb_P_R"),

                                                            ("RightHandIndex4", "index_EF_R"),
                                                            ("RightHandIndex3", "index_D_R"),
                                                            ("RightHandIndex2", "index_M_R"),
                                                            ("RightHandIndex1", "index_P_R"),

                                                            ("RightHandMiddle4", "middle_EF_R"),
                                                            ("RightHandMiddle3", "middle_D_R"),
                                                            ("RightHandMiddle2", "middle_M_R"),
                                                            ("RightHandMiddle1", "middle_P_R"),

                                                            ("RightHandRing4", "ring_EF_R"),
                                                            ("RightHandRing3", "ring_D_R"),
                                                            ("RightHandRing2", "ring_M_R"),
                                                            ("RightHandRing1", "ring_P_R"),

                                                            ("RightHandPinky4", "pinky_EF_R"),
                                                            ("RightHandPinky3", "pinky_D_R"),
                                                            ("RightHandPinky2", "pinky_M_R"),
                                                            ("RightHandPinky1", "pinky_P_R"),

                                                            ("LeftHandThumb4", "thumb_EF_L"),
                                                            ("LeftHandThumb3", "thumb_D_L"),
                                                            ("LeftHandThumb2", "thumb_P_L"),

                                                            ("LeftHandIndex4", "index_EF_L"),
                                                            ("LeftHandIndex3", "index_D_L"),
                                                            ("LeftHandIndex2", "index_M_L"),
                                                            ("LeftHandIndex1", "index_P_L"),

                                                            ("LeftHandMiddle4", "middle_EF_L"),
                                                            ("LeftHandMiddle3", "middle_D_L"),
                                                            ("LeftHandMiddle2", "middle_M_L"),
                                                            ("LeftHandMiddle1", "middle_P_L"),

                                                            ("LeftHandRing4", "ring_EF_L"),
                                                            ("LeftHandRing3", "ring_D_L"),
                                                            ("LeftHandRing2", "ring_M_L"),
                                                            ("LeftHandRing1", "ring_P_L"),

                                                            ("LeftHandPinky4", "pinky_EF_L"),
                                                            ("LeftHandPinky3", "pinky_D_L"),
                                                            ("LeftHandPinky2", "pinky_M_L"),
                                                            ("LeftHandPinky1", "pinky_P_L"),

                                                            ("Skeleton 001:RWristOut", "hand_radius_R"),
                                                            ("Skeleton 001_RHand", "wrist_R"),
                                                            ("Skeleton 001_RFArm", "elbow_R"),

                                                            ("Skeleton 001:LWristOut", "hand_radius_L"),
                                                            ("Skeleton 001_LHand", "wrist_L"),
                                                            ("Skeleton 001_LFArm", "elbow_L")],

                                           "functions": [(["Skeleton 001:RWristIn", "Skeleton 001:RHandOut"], "palm_R", "palm"),
                                                         (["Skeleton 001:LWristIn", "Skeleton 001:LHandOut"], "palm_L", "palm")]}

identifier_columns = ["Id", "Label"]
identifier_columns = ["Id", "Label", "Frame"]

running_info = {# "retrieving_mode": "average",
                # "retrieving_mode": "all",
                "retrieving_mode": "duration",
                # "hand_mode": "both",
                "hand_mode": "one",
                "correspondence_mocap_kinematic_variable": correspondence_mocap_kinematic_variable,
                "identifier_columns": identifier_columns}
#----------------------------------------------------------------------------------------------------------------------#


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, running_info=running_info)
manager.run()
