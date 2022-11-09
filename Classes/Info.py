from tabulate import tabulate
import os, itertools
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
import json

class PathInfo:

    def __init__(self):

        """
        that class is used to get all information relative to path (Data, DB, figure...)
        """

        # Get current directory path
        user_name = os.getlogin()
        if user_name == 'GVLAB':
            self.cwd = os.getcwd()
            self.path_parent_project = "D:\\AMADA"
        else:
            self.cwd = os.getcwd()
            self.path_parent_project = os.path.abspath(os.path.join(self.cwd, os.pardir))

        # Get path of one level above the root of the project

        # Get path of the data as experiment_1 to experiment_5
        self.path_experiment_1 = self.path_parent_project + '\\experiment_1'
        self.path_experiment_2 = self.path_parent_project + '\\experiment_2'
        self.path_experiment_3 = self.path_parent_project + '\\experiment_3'
        self.path_experiment_4 = self.path_parent_project + '\\experiment_4'
        self.path_experiment_5 = self.path_parent_project + '\\experiment_5'

        print('cwd : {0}'.format(self.cwd))
        print('path of the parent project : {0}'.format(self.path_parent_project))
        print('path of the file as the experiment_1 : {0}'.format(self.path_experiment_1))
        print('path of the file as the experiment_2 : {0}'.format(self.path_experiment_2))
        print('path of the file as the experiment_3 : {0}'.format(self.path_experiment_3))
        print('path of the file as the experiment_4 : {0}'.format(self.path_experiment_4))
        print('path of the file as the experiment_5 : {0}'.format(self.path_experiment_5))

        self.path_encoded_json = self.path_parent_project + "\\Json"


class IDataInfo(ABC):

    def __init__(self):

        self._set_pathinfo()

    @abstractmethod
    def _set_pathinfo(self):

        self.pathInfo = PathInfo()

        self.path_experiment = ""

    @abstractmethod
    def get_experiment_info(self):

        # analysis paths
        analysis_path = self.path_experiment + "\\analysis\\"
        bendinfo_path = analysis_path + "bendinfo\\"
        bendinfo_bend_ids = [folder for folder in os.listdir(bendinfo_path)
                             if os.path.isdir(os.path.join(bendinfo_path, folder))]
        bendinfo_bend_paths = [bendinfo_path + folder + "\\" for folder in bendinfo_bend_ids]
        information_path = analysis_path + "information\\"

        if not (os.path.exists(analysis_path)):
            os.makedirs(analysis_path)
        if not (os.path.exists(bendinfo_path)):
            os.makedirs(bendinfo_path)
        if not (os.path.exists(information_path)):
            os.makedirs(information_path)

        self.experiment_path_dict = {"analysis_dir": analysis_path,
                                     "bendinfo_dir": bendinfo_path,
                                     "bendinfo_bend_ids": bendinfo_bend_ids,
                                     "bendinfo_bend_paths": bendinfo_bend_paths,
                                     "information_dir": information_path,
                                     "bend_info": information_path + "bend_info.csv",
                                     "take_info": information_path + "take_info.json",
                                     "velocity_cheat_info": information_path + "velocity_cheat_info.txt",
                                     "position_cheat_info": information_path + "position_cheat_info.txt",
                                     "grasp_info": information_path + "grasp_info.json",
                                     "dendrogram_csv": information_path + "dendrogram.csv"}

    @abstractmethod
    def set_data_info(self, name_take: str):

        self.path_encoded_json = self.pathInfo.path_parent_project + "\\Json"

        # get condition.json
        condition_file = open(self.path_experiment + "\\" + name_take + "\\result\\" + "condition.json", "r")
        condition_dict = json.load(condition_file)

        # analysis paths
        analysis_path = self.path_experiment + "\\analysis\\"
        bendinfo_path = analysis_path + "bendinfo\\"
        bendinfo_bend_ids = [folder for folder in os.listdir(bendinfo_path)
                             if os.path.isdir(os.path.join(bendinfo_path, folder))]
        bendinfo_bend_paths = [bendinfo_path + folder + "\\" for folder in bendinfo_bend_ids]
        information_path = analysis_path + "information\\"

        # result_path = self.path_experiment + "\\" + name_take + "\\org" + "\\" + condition_dict["timestamp"] + "\\データ\\"
        result_path = self.path_experiment + "\\" + name_take + "\\result\\"

        velocity_path = result_path + "velocity\\"

        acceleration_path = result_path + "acceleration\\"

        jerk_path = result_path + "jerk\\"

        path_indices = result_path + "indices\\"

        grasp_path = result_path + "grasp\\"

        # extraction
        path_extraction = result_path + "extraction\\"
        path_extraction_original = path_extraction + "original\\"
        path_extraction_velocity = path_extraction + "velocity\\"
        path_extraction_acceleration = path_extraction + "acceleration\\"
        path_extraction_jerk = path_extraction + "jerk\\"

        if not (os.path.exists(analysis_path)):
            os.makedirs(analysis_path)
        if not (os.path.exists(bendinfo_path)):
            os.makedirs(bendinfo_path)
        if not (os.path.exists(information_path)):
            os.makedirs(information_path)
        if not (os.path.exists(path_indices)):
            os.makedirs(path_indices)
        if not (os.path.exists(velocity_path)):
            os.makedirs(velocity_path)
        if not (os.path.exists(acceleration_path)):
            os.makedirs(acceleration_path)
        if not (os.path.exists(jerk_path)):
            os.makedirs(jerk_path)
        if not (os.path.exists(grasp_path)):
            os.makedirs(grasp_path)
        # extraction
        if not (os.path.exists(path_extraction)):
            os.makedirs(path_extraction)
        if not (os.path.exists(path_extraction_original)):
            os.makedirs(path_extraction_original)
        if not (os.path.exists(path_extraction_velocity)):
            os.makedirs(path_extraction_velocity)
        if not (os.path.exists(path_extraction_acceleration)):
            os.makedirs(path_extraction_acceleration)
        if not (os.path.exists(path_extraction_jerk)):
            os.makedirs(path_extraction_jerk)

        self.path_dict = {}

        self.path_dict["original"] = {"name": "original",
                                      "data": result_path + name_take + "_mm_NANSENSEMerged_M.csv",
                                      "result": result_path,
                                      "result_v": velocity_path,
                                      "result_a": acceleration_path,
                                      "result_j": jerk_path,
                                      "indices": path_indices}

        self.path_dict["velocity"] = {"name": "velocity",
                                      "data": velocity_path + "velocity.csv",
                                      "result_v": velocity_path,
                                      "result_a": acceleration_path,
                                      "result_j": jerk_path,
                                      "result": velocity_path}

        self.path_dict["acceleration"] = {"name": "acceleration",
                                          "data": acceleration_path + "acceleration.csv",
                                          "result_v": velocity_path,
                                          "result_a": acceleration_path,
                                          "result_j": jerk_path,
                                          "result": acceleration_path}

        self.path_dict["jerk"] = {"name": "jerk",
                                  "data": jerk_path + "jerk.csv",
                                  "result_v": velocity_path,
                                  "result_a": acceleration_path,
                                  "result_j": jerk_path,
                                  "result": jerk_path}

        self.path_dict["segment_DB"] = {"name": "segment_DB",
                                        "data_x": result_path + "DB_X.npy",
                                        "data_y": result_path + "DB_Y.npy",
                                        "std": result_path + "DB_std.npy",
                                        "labels": result_path + "labels.npy",
                                        "result": result_path}

        self.path_dict["indices"] = {"name": "indices",
                                     "data": path_indices}

        self.path_dict["extraction"] = {"name": "extraction",
                                        "data": result_path + name_take + "_mm_NANSENSEMerged_M.csv",
                                        "result": path_extraction_original}

        self.path_dict["extracted_original"] = {"name": "extracted_original",
                                                "data": path_extraction_original + name_take + ".csv",
                                                "result_v": path_extraction_velocity,
                                                "result_a": path_extraction_acceleration,
                                                "result_j": path_extraction_jerk}

        self.path_dict["grasp_original"] = {"name": "grasp_original",
                                            "data": result_path + name_take + "_mm_NANSENSEMerged_M.csv",
                                            "result": grasp_path,
                                            "velocity": grasp_path + "velocity.csv",
                                            "time": analysis_path + "grasp_time.csv",
                                            "take_info": information_path + "take_info.json"}

        self.path_dict["grasp_velocity"] = {"name": "grasp_velocity",
                                            "data": grasp_path + "velocity.csv",
                                            "result": grasp_path,
                                            "average": grasp_path + "average.csv",
                                            "all": grasp_path + "all.csv",
                                            "point": grasp_path + "point.csv",
                                            "duration": grasp_path + "duration.csv",
                                            "take_info": information_path + "take_info.json"}

        self.path_dict["grasp_retrieving"] = {"name": "",
                                              "posture_name": "grasp_retrieving_posture",
                                              "motion_name": "grasp_retrieving_motion",
                                              "data": "",
                                              "average_data": grasp_path + "average.csv",
                                              "all_data": grasp_path + "all.csv",
                                              "point_data": grasp_path + "point.csv",
                                              "duration_data": grasp_path + "duration.csv",
                                              "bend_info_csv": information_path + "bend_info.csv",
                                              "result": "",
                                              "average_result": grasp_path + "labeling_average.csv",
                                              "all_result": grasp_path + "labeling_all.csv",
                                              "point_result": grasp_path + "labeling_point.csv",
                                              "duration_result": grasp_path + "labeling_duration.csv",
                                              "take_info": information_path + "take_info.json"}

        self.path_dict["grasp_labeling"] = {"name": "",
                                            "posture_name": "grasp_labeling_posture",
                                            "motion_name": "grasp_labeling_motion",
                                            "data": "",
                                            "average_data": grasp_path + "labeling_average.csv",
                                            "all_data": grasp_path + "labeling_all.csv",
                                            "point_data": grasp_path + "labeling_point.csv",
                                            "duration_data": grasp_path + "labeling_duration.csv",
                                            "result": analysis_path,
                                            "dataset": "",
                                            "average_dataset": analysis_path + "dataset_average_mocap.csv",
                                            "all_dataset": analysis_path + "dataset_all_mocap.csv",
                                            "point_dataset": analysis_path + "dataset_point_mocap.csv",
                                            "duration_dataset": analysis_path + "dataset_duration_mocap.csv",
                                            "take_info": information_path + "take_info.json"}

        self.path_dict["grasp_dataset_mocap_average"] = {"name": "grasp_dataset_mocap_average",
                                                         "data": analysis_path + "dataset_average_mocap.csv",
                                                         "result": analysis_path,
                                                         "kinematic": analysis_path + "dataset_average_kinematic_variable.csv"}

        self.path_dict["grasp_dataset_mocap_all"] = {"name": "grasp_dataset_mocap_all",
                                                     "data": analysis_path + "dataset_all_mocap.csv",
                                                     "result": analysis_path,
                                                     "kinematic": analysis_path + "dataset_all_kinematic_variable.csv"}

        self.path_dict["grasp_dataset_mocap_duration"] = {"name": "grasp_dataset_mocap_duration",
                                                     "data": analysis_path + "dataset_duration_mocap.csv",
                                                     "result": analysis_path,
                                                     "kinematic": analysis_path + "dataset_duration_kinematic_variable.csv"}

        self.path_dict["grasp_cleansing_average"] = {"name": "grasp_cleansing_average",
                                                     "data": analysis_path + "average.csv",
                                                     "result": analysis_path,
                                                     "cleansing": analysis_path + "cleansing_average.csv"}

        self.path_dict["grasp_cleansing_all"] = {"name": "grasp_cleansing_all",
                                                 "data": analysis_path + "all.csv",
                                                 "result": analysis_path,
                                                 "cleansing": analysis_path + "cleansing_all.csv"}

        self.path_dict["analysis"] = {"name": "analysis",
                                      "folder": analysis_path,
                                      "bendinfo": bendinfo_path,
                                      "bendinfo_bends": bendinfo_bend_paths,
                                      "bendinfo_bend_ids": bendinfo_bend_ids,
                                      "bendinfo_csv": analysis_path + "bend_info.csv"}

    @abstractmethod
    def get_data_dict(self, name_file: str):

        self.name_file = name_file

        if self.name_file in self.path_dict:
            return self.path_dict[self.name_file]

        else:
            return None


class DataInfo1(IDataInfo):

    def __init__(self):

        super().__init__()

    def _set_pathinfo(self):

        super()._set_pathinfo()

        self.path_experiment = self.pathInfo.path_experiment_1

    def get_experiment_info(self):

        super().get_experiment_info()

        return self.experiment_path_dict

    def set_data_info(self, name_take: str):

        super().set_data_info(name_take)

    def get_data_dict(self, name_file: str):

        self.data_dict = super().get_data_dict(name_file)

        return self.data_dict


class DataInfo2(IDataInfo):

    def __init__(self):

        super().__init__()

    def _set_pathinfo(self):

        super()._set_pathinfo()

        self.path_experiment = self.pathInfo.path_experiment_2

    def get_experiment_info(self):

        super().get_experiment_info()

        return self.experiment_path_dict

    def set_data_info(self, name_take: str):

        super().set_data_info(name_take)

    def get_data_dict(self, name_file: str):

        self.data_dict = super().get_data_dict(name_file)

        return self.data_dict


class DataInfo3(IDataInfo):

    def __init__(self):

        super().__init__()

    def _set_pathinfo(self):

        super()._set_pathinfo()

        self.path_experiment = self.pathInfo.path_experiment_3

    def get_experiment_info(self):

        super().get_experiment_info()

        return self.experiment_path_dict

    def set_data_info(self, name_take: str):

        super().set_data_info(name_take)

    def get_data_dict(self, name_file: str):

        self.data_dict = super().get_data_dict(name_file)

        return self.data_dict


class DataInfo4(IDataInfo):

    def __init__(self):

        super().__init__()

    def _set_pathinfo(self):

        super()._set_pathinfo()

        self.path_experiment = self.pathInfo.path_experiment_4

    def get_experiment_info(self):

        super().get_experiment_info()

        return self.experiment_path_dict

    def set_data_info(self, name_take: str):

        super().set_data_info(name_take)

    def get_data_dict(self, name_file: str):

        self.data_dict = super().get_data_dict(name_file)

        return self.data_dict


class DataInfo5(IDataInfo):

    def __init__(self):

        super().__init__()

    def _set_pathinfo(self):

        super()._set_pathinfo()

        self.path_experiment = self.pathInfo.path_experiment_5

    def get_experiment_info(self):

        super().get_experiment_info()

        return self.experiment_path_dict

    def set_data_info(self, name_take: str):

        super().set_data_info(name_take)

    def get_data_dict(self, name_file: str):

        self.data_dict = super().get_data_dict(name_file)

        return self.data_dict
