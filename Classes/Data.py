import math, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from glob import glob

from typing import List, Dict
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from Classes.Info import PathInfo, DataInfo1, IDataInfo, DataInfo2


class DataFactory():

    def __init__(self, data_dict: dict, data_info: IDataInfo):

        self.data_dict = data_dict
        self.data_info = data_info

        if "original" == self.data_dict["name"]:

            self.dataStrategy = CSVDataStrategy()

        elif "velocity" == self.data_dict["name"] or "acceleration" == self.data_dict["name"] or "jerk" == self.data_dict["name"]:

            self.dataStrategy = CSVDataVAStrategy()

        elif "segment_DB" == self.data_dict["name"]:

            self.dataStrategy = NumpySegmentDBStrategy()

        elif "indices" == self.data_dict["name"]:

            self.dataStrategy = NumpyAllIndicesStrategy()

        elif "extraction" == self.data_dict["name"]:

            self.dataStrategy = CSVExtractionStrategy()

        elif "extracted_original" == self.data_dict["name"]:

            self.dataStrategy = CSVExtractedDataStrategy()

        elif "grasp" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspStrategy()
        
        elif "grasp_original" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspOriginalStrategy()

        elif "grasp_velocity" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspVelocityStrategy()

        elif "grasp_retrieving_posture" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspRetrievingPostureStrategy()

        elif "grasp_retrieving_motion" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspRetrievingMotionStrategy()

        elif "grasp_labeling_posture" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspLabelingPostureStrategy()

        elif "grasp_labeling_motion" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspLabelingMotionStrategy()

        elif "grasp_cleansing_average" == self.data_dict["name"] or "grasp_cleansing_all" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspLabelingPostureStrategy()

        elif "grasp_dataset_mocap_average" == self.data_dict["name"] or "grasp_dataset_mocap_all" == self.data_dict["name"]\
                or "grasp_dataset_mocap_duration" == self.data_dict["name"]:

            self.dataStrategy = CSVGraspDatasetStrategy()

        else:
            pass

    def create(self):

        self.data = self.dataStrategy.get_data(data_dict=self.data_dict)

        return self.data


class IData(ABC):

    data_dict: dict

    @abstractmethod
    def get_data(self, data_dict):
        pass


class CSVDataStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6])
        df_value.rename(columns={df_value.columns.values[0]: "Frame"}, inplace=True)
        df_value.rename(columns={df_value.columns.values[1]: "Time"}, inplace=True)

        df_properties = pd.read_csv(path_dict["data"], sep=",", header=1, nrows=4)
        df_properties.rename(columns={df_properties.columns.values[0]: "Frame"}, inplace=True)
        df_properties.rename(columns={df_properties.columns.values[1]: "Time"}, inplace=True)

        return df_value, df_properties


class CSVDataVAStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[1, 2, 3, 4])
        df_value.rename(columns={df_value.columns.values[0]: "Frame"}, inplace=True)
        df_value.rename(columns={df_value.columns.values[1]: "Time"}, inplace=True)

        df_properties = pd.read_csv(path_dict["data"], sep=",", header=0, nrows=4)
        df_properties.rename(columns={df_properties.columns.values[0]: "Frame"}, inplace=True)
        df_properties.rename(columns={df_properties.columns.values[1]: "Time"}, inplace=True)

        return df_value, df_properties


class NumpySegmentDBStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict
        # print("the path of data called from file : {0}".format(path_dict["data"]))

        DB_X = np.load(path_dict["data_x"])
        DB_Y = np.load(path_dict["data_y"])
        DB_std = np.load(path_dict["std"])
        labels = np.load(path_dict["labels"])

        return DB_X, DB_Y, DB_std, labels


class NumpyAllIndicesStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        # print("the path of data called from file : {0}".format(path_dict["data"]))

        dict_indices = {}

        for file in glob(path_dict["data"] + "/*.npy"):
            indices_segments = np.load(file)
            dict_indices[len(indices_segments)] = indices_segments

            print("The number of segments is {0}".format(len(indices_segments)))

        print("The length of dictionary is {0}".format(len(dict_indices)))

        return dict_indices


class CSVExtractionStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 2, 3, 4, 5], usecols=lambda x: x not in ['Frame', 'Time'], skip_blank_lines=False)

        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)

        df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)

        df_timeSeries = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_timeSeries


class CSVExtractedDataStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6])
        df_value.rename(columns={df_value.columns.values[0]: "Frame"}, inplace=True)
        df_value.rename(columns={df_value.columns.values[1]: "Time"}, inplace=True)

        df_properties = pd.read_csv(path_dict["data"], sep=",", header=2, nrows=4)
        df_properties.rename(columns={df_properties.columns.values[0]: "Frame"}, inplace=True)
        df_properties.rename(columns={df_properties.columns.values[1]: "Time"}, inplace=True)

        return df_value, df_properties


class CSVGraspStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        if os.path.exists(path_dict["velocity"]) is False:

            print("the path of data called from file : {0}".format(path_dict["data"]))

            df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 2, 3, 4, 5], usecols=lambda x: x not in ['Frame', 'Time'], skip_blank_lines=False)
            df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
            df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)
            df_timeSeries = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        else:

            print("the path of data called from file : {0}".format(path_dict["velocity"]))

            df_value = pd.read_csv(path_dict["velocity"], sep=",", skiprows=[0, 1, 3, 4, 5, 6], usecols=lambda x: x not in ['Frame', 'Time'], skip_blank_lines=False)
            # REVIEW: Some changes may be necessary as above
            df_properties = pd.read_csv(path_dict["velocity"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
            df_header = pd.read_csv(path_dict["velocity"], sep=",", header=None, nrows=2, skip_blank_lines=False)
            df_timeSeries = pd.read_csv(path_dict["velocity"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_timeSeries


class CSVGraspOriginalStrategy(IData):
    
    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 2, 3, 4, 5], usecols=lambda x: x not in ['Frame', 'Time'], skip_blank_lines=False).fillna(0.0)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)
        df_timeSeries = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_timeSeries


# TODO: Make this class the same as the above class
class CSVGraspVelocityStrategy(IData):
    
    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 2, 3, 4, 5], usecols=lambda x: x not in ['Frame', 'Time'], skip_blank_lines=False)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)
        df_timeSeries = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_timeSeries


class CSVGraspRetrievingPostureStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6], usecols=lambda x: x not in ['Id', 'Name'], skip_blank_lines=False)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)
        df_identifier = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_identifier


class CSVGraspRetrievingMotionStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6], usecols=lambda x: x not in ['Id', 'Name', 'Frame'], skip_blank_lines=False)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_header = pd.read_csv(path_dict["data"], sep=",", header=None, nrows=2, skip_blank_lines=False)
        df_identifier = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1, 2], skip_blank_lines=False)

        return df_value, df_properties, df_header, df_identifier


class CSVGraspLabelingPostureStrategy(IData):
    
    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6], usecols=lambda x: x not in ['Id', 'Label'], skip_blank_lines=False)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_identifier = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1], skip_blank_lines=False)

        return df_value, df_properties, df_identifier


class CSVGraspLabelingMotionStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df_value = pd.read_csv(path_dict["data"], sep=",", skiprows=[0, 1, 3, 4, 5, 6], usecols=lambda x: x not in ['Id', 'Label', 'Frame'], skip_blank_lines=False)
        df_properties = pd.read_csv(path_dict["data"], sep=",", header=None, skiprows=lambda x: x not in [2, 3, 4, 5, 6], skip_blank_lines=False)
        df_identifier = pd.read_csv(path_dict["data"], sep=",", header=0, skiprows=6, usecols=[0, 1, 2], skip_blank_lines=False)

        return df_value, df_properties, df_identifier


class CSVGraspDatasetStrategy(IData):

    def get_data(self, data_dict):

        path_dict = data_dict

        print("the path of data called from file : {0}".format(path_dict["data"]))

        df = pd.read_csv(path_dict["data"], sep=",")

        return df


class Creation():

    def __init__(self, df_original: tuple, data_dict:dict):

        path_dict = data_dict

        # call the path for saving the result
        self.result_path_v = path_dict["result_v"]
        self.result_path_a = path_dict["result_a"]
        self.result_path_j = path_dict["result_j"]
        self.data_path = path_dict["data"]

        # call the data frame data from tuple
        df = df_original
        self.value = df[0]
        self.properties = df[1]

    def get_CSV(self):

        # take the second derivative for calculating acc
        self.df = self.derivative()
        self.df_vel = self.df[0]
        self.df_acc = self.df[1]

        # save csv file
        self.save_csv_vel(df_data=self.df_vel, result_path=self.result_path_v)
        self.save_csv_acc(df_data=self.df_acc, result_path=self.result_path_a)

        print("Computing velocity...")
        df_diff = Creation.differential_ndarray(df_data=self.value)
        print("Have computed velocity!")
        Creation.save_csv_vel(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                              result_path=self.result_path_v)

        print("Computing acceleration...")
        df_diff = Creation.differential_ndarray(df_data=df_diff)
        print("Have computed acceleration!")
        Creation.save_csv_acc(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                              result_path=self.result_path_a)

        print("Computing jerk...")
        df_diff = Creation.differential_ndarray(df_data=df_diff)
        print("Have computed acceleration!")
        Creation.save_csv_jerk(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                               result_path=self.result_path_j)

    def derivative(self):

        # print("while calculating speed.")
        # v_value = self.differential(df_data=self.value)
        #
        # print("while calculating acceleration.")
        # a_value = self.differential(df_data=v_value)

        print("Computing velocity...")
        v_value = self.differential_ndarray(df_data=self.value)
        print("Have computed velocity!")

        print("Computing acceleration...")
        a_value = self.differential_ndarray(df_data=v_value)
        print("Have computed acceleration!")

        df_vel = pd.concat([self.properties, v_value], axis=0, ignore_index=True)

        df_acc = pd.concat([self.properties, a_value], axis=0, ignore_index=True)

        return df_vel, df_acc

    @staticmethod
    def differential(df_data):
        # calculation for acceleration
        df = pd.DataFrame()
        times = [df_data.iloc[i, 1] for i in df_data.iloc[:, 1].index.values[:]]
        other = df_data.iloc[:-1, 0:2]

        for i in tqdm(range(len(times) - 1)):

            component_1 = df_data.iloc[[i], 2:]
            component_2 = df_data.iloc[[i + 1], 2:]
            component_3 = times[i]
            component_4 = times[i + 1]

            numerator = component_2 - component_1.values
            denominator = component_4 - component_3

            results = numerator / denominator

            df = pd.concat([df, results], axis=0, ignore_index=True)

        df = pd.concat([other, df], axis=1)

        return df

    @staticmethod
    def differential_ndarray(df_data):

        values = df_data.values

        column_frames = values[:, 0]
        column_times = values[:, 1]

        values_i = values
        values_i1 = np.delete(values_i, 0, axis=0)
        values_i = np.delete(values_i, -1, axis=0)

        values_delta = values_i1 - values_i

        values_differential = values_delta[:, 2:] / values_delta[:, 1].reshape(-1, 1)

        values_result = np.round(values_differential, decimals=6)

        values_result = np.concatenate([column_frames[:-1].reshape(-1, 1), column_times[:-1].reshape(-1, 1), values_result], axis=1)

        df_diff = pd.DataFrame(data=values_result, index=df_data.index.values[:-1], columns=df_data.columns)

        return df_diff

    @staticmethod
    def save_csv_jerk(df_data: pd.DataFrame, result_path: str):

        name = "jerk.csv"
        path_name = result_path + name

        print("Saving jerk csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved jerk csv!")

    @staticmethod
    def save_csv_acc(df_data: pd.DataFrame, result_path: str):

        name = "acceleration.csv"
        path_name = result_path + name

        print("Saving acceleration csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved acceleration csv!")

    @staticmethod
    def save_csv_vel(df_data: pd.DataFrame, result_path: str):

        name = "velocity.csv"
        path_name = result_path + name

        print("Saving velocity csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved velocity csv!")


# TODO: I want to integrate this class to Creation class (by Takumi)
class Creation_Extracted():

    def __init__(self, df_original_extracted: tuple, data_dict: dict, target_params_dict: Dict[str, bool]):

        path_dict = data_dict

        # call the path for saving the result
        self.result_path_v = path_dict["result_v"] if target_params_dict["velocity"] else None
        self.result_path_a = path_dict["result_a"] if target_params_dict["acceleration"] else None
        self.result_path_j = path_dict["result_j"] if target_params_dict["jerk"] else None
        self.data_path = path_dict["data"]

        self.last_param = None

        for param in ["jerk", "acceleration", "velocity"]:

            if target_params_dict[param]:

                self.last_param = param
                break

        # call the data frame data from tuple
        df = df_original_extracted
        self.value = df[0]
        self.properties = df[1]

    def get_CSV(self, extracted_file_neme: str = ""):

        if self.last_param is None:
            return

        print("Computing velocity...")
        df_diff = Creation.differential_ndarray(df_data=self.value)
        print("Have computed velocity!")
        if self.result_path_v is not None:
            Creation_Extracted.save_csv_vel(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                                            result_path=self.result_path_v + extracted_file_neme)

        if self.last_param == "velocity":
            return

        print("Computing acceleration...")
        df_diff = Creation.differential_ndarray(df_data=df_diff)
        print("Have computed acceleration!")
        if self.result_path_a is not None:
            Creation_Extracted.save_csv_acc(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                                            result_path=self.result_path_a + extracted_file_neme)

        if self.last_param == "acceleration":
            return

        print("Computing jerk...")
        df_diff = Creation.differential_ndarray(df_data=df_diff)
        print("Have computed acceleration!")
        if self.result_path_j is not None:
            Creation_Extracted.save_csv_jerk(df_data=pd.concat([self.properties, df_diff], axis=0, ignore_index=True),
                                             result_path=self.result_path_j + extracted_file_neme)

        if self.last_param == "jerk":
            return

    @staticmethod
    def differential_ndarray(df_data):

        values = df_data.values

        column_frames = values[:, 0]
        column_times = values[:, 1]

        values_i = values
        values_i1 = np.delete(values_i, 0, axis=0)
        values_i = np.delete(values_i, -1, axis=0)

        values_delta = values_i1 - values_i

        values_differential = values_delta[:, 2:] / values_delta[:, 1].reshape(-1, 1)

        values_result = np.round(values_differential, decimals=6)

        values_result = np.concatenate([column_frames[:-1].reshape(-1, 1), column_times[:-1].reshape(-1, 1), values_result], axis=1)

        df_diff = pd.DataFrame(data=values_result, index=df_data.index.values[:-1], columns=df_data.columns)

        return df_diff

    @staticmethod
    def save_csv_jerk(df_data: pd.DataFrame, result_path: str):

        name = "_jerk.csv"
        path_name = result_path + name

        print("Saving jerk csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved jerk csv!")

    @staticmethod
    def save_csv_acc(df_data: pd.DataFrame, result_path: str):

        name = "_acceleration.csv"
        path_name = result_path + name

        print("Saving acceleration csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved acceleration csv!")

    @staticmethod
    def save_csv_vel(df_data: pd.DataFrame, result_path: str):

        name = "_velocity.csv"
        path_name = result_path + name

        print("Saving velocity csv...")
        df_data.to_csv(path_name, index=False)
        print("Have saved velocity csv!")

class Creation_Grasp():

    def __init__(self, df_original: tuple, data_dict: dict):

        path_dict = data_dict

        # call the path for saving the result
        self.velocity_path = path_dict["velocity"]
        self.data_path = path_dict["data"]

        # call the data frame data from tuple
        df = df_original
        self.value = df[0]
        self.properties = df[1]
        self.header = df[2]

    def get_CSV(self):

        print("Computing velocity...")
        df_diff = self.differential_ndarray(df_data=self.value)
        print("Have computed velocity!")

        target_df = pd.concat([self.properties, df_diff], axis=0, ignore_index=True)
        target_df = pd.concat([self.header, target_df], axis=0, ignore_index=True)
        self.save_csv_vel(df_data=target_df, result_path=self.velocity_path)


    @staticmethod
    def differential_ndarray(df_data):

        values = df_data.values

        column_frames = values[:, 0]
        column_times = values[:, 1]

        values_i = values
        values_i1 = np.delete(values_i, 0, axis=0)
        values_i = np.delete(values_i, -1, axis=0)

        values_delta = values_i1 - values_i

        values_differential = values_delta[:, 2:] / values_delta[:, 1].reshape(-1, 1)

        values_result = np.round(values_differential, decimals=6)

        values_result = np.concatenate([column_frames[:-1].reshape(-1, 1), column_times[:-1].reshape(-1, 1), values_result], axis=1)

        df_diff = pd.DataFrame(data=values_result, index=df_data.index.values[:-1], columns=df_data.columns)

        return df_diff

    @staticmethod
    def save_csv_vel(df_data: pd.DataFrame, result_path: str):

        path_name = result_path

        print("Saving velocity csv...")
        df_data.to_csv(path_name, index=False, header=False)
        print("Have saved velocity csv!")


class DataReader():

    """
    example for each param.

    bone_param = {
        'flag': True,
        'parts': ["Skeleton 001_LFoot","Skeleton 001_RThigh"],
        'position': ["X"],
        'rotation': ["W", "Z", "Y", "X"]
    }
    rigid_body_param = {
        'flag': True,
        'parts': ["LeftHandIndex","LeftHandIndex1"],
        'position': ["X","Y","Z"],
        'rotation': ["W", "Z", "Y"]
    }
    marker_param = {
        'flag': True,
        'parts': ["GazeDirectionL","GazeDirectionR"],
        'position': ["X","Y","Z"],
        'rotation': []
    }
    """

    def __init__(self, values: pd.DataFrame, properties: pd.DataFrame, bone: dict, rigid_body: dict, marker: dict,
                 exists_properties_columns: bool = True):

        self.data = values, properties

        self.exists_properties_columns = exists_properties_columns

        dataValues, dataProperties = pd.DataFrame(), pd.DataFrame()

        if bone['flag']:
            data = self.unitsPick(bone, 'Bone')
            dataValues = pd.concat([dataValues, data[0]], axis=1)
            dataProperties = pd.concat([dataProperties, data[1]], axis=1)

        if rigid_body['flag']:
            data = self.unitsPick(rigid_body, 'Rigid Body')
            dataValues = pd.concat([dataValues, data[0]], axis=1)
            dataProperties = pd.concat([dataProperties, data[1]], axis=1)

        if marker['flag']:
            data = self.unitsPick(marker, 'Marker')
            dataValues = pd.concat([dataValues, data[0]], axis=1)
            dataProperties = pd.concat([dataProperties, data[1]], axis=1)

        self.final_data = dataValues, dataProperties

    def getData(self):
        return self.final_data

    def unitsPick(self, unit: dict, unit_name: str):
        # index_units = self.getUnitsIndex(data=self.data, units=unit_name)
        index_units = self.getUnitsIndex(units=unit_name)
        index_parts = np.empty((0, 7), dtype=int)
        if not unit['parts'] == []:
            for pa in unit['parts']:
                p = self.getPartsIndex(unitsIndex=index_units, parts=pa)
                index_parts = np.append(index_parts, [p], axis=0)
                # print(index_parts)
        else:
            index_parts = np.append(index_parts, self.getPartsIndex(unitsIndex=index_units, parts=None), axis=0)
        # print(index_parts)
        index_pos_rot = np.empty((0, 7), dtype=int)
        for ip in index_parts:
            index_pos_rot = np.append(index_pos_rot,
                                      [self.getPosAndRotIndex(partsIndex=ip, position=unit['position'],
                                                              rotation=unit['rotation'])], axis=0)
        # print(index_pos_rot)
        dataValues, dataProperties = pd.DataFrame(), pd.DataFrame()
        for ipr in index_pos_rot:
            d = self.pick(index_pos_rot=ipr, position=unit['position'], rotation=unit['rotation'])
            dataValues = pd.concat([dataValues, d[0]], axis=1)
            dataProperties = pd.concat([dataProperties, d[1]], axis=1)
        return dataValues, dataProperties

    def pick(self, index_pos_rot, position: list, rotation: list):

        # print("while getting data ...")
        index = index_pos_rot

        if position == None and rotation == None:
            data = self.callData(self.data, index_pos_rot)
            return data

        pos_finalIndex = np.empty(0, dtype=int)
        rot_finalIndex = np.empty(0, dtype=int)

        for pos in position:

            if "X" == pos:
                pos_finalIndex = np.append(pos_finalIndex, index_pos_rot[0])
            if "Y" == pos:
                pos_finalIndex = np.append(pos_finalIndex, index_pos_rot[1])
            if "Z" == pos:
                pos_finalIndex = np.append(pos_finalIndex, index_pos_rot[2])

        for rot in rotation:

            if "X" == rot:
                rot_finalIndex = np.append(rot_finalIndex, index_pos_rot[3])
            if "Y" == rot:
                rot_finalIndex = np.append(rot_finalIndex, index_pos_rot[4])
            if "Z" == rot:
                rot_finalIndex = np.append(rot_finalIndex, index_pos_rot[5])
            if "W" == rot:
                rot_finalIndex = np.append(rot_finalIndex, index_pos_rot[6])

        finalIndex = np.append(pos_finalIndex, rot_finalIndex)
        data = self.callData(data=self.data, index=finalIndex)
        return data

    @staticmethod
    def callData(data, index: np.ndarray):

        finalValues = pd.DataFrame()  # 空のdataFrame
        finalProperties = pd.DataFrame()

        for n in index:

            d = data[0].iloc[:, n]
            p = data[1].iloc[:, n]

            finalValues = pd.concat([finalValues, d], axis=1)
            finalProperties = pd.concat([finalProperties, p], axis=1)

        return finalValues, finalProperties

    def getPosAndRotIndex(self, partsIndex: bytearray, position: list, rotation: list):

        # print("while getting data pos and rot index ...")

        if position == None and rotation == None:

            return partsIndex

        p_index_X = 0
        p_index_Y = 0
        p_index_Z = 0

        r_index_X = 0
        r_index_Y = 0
        r_index_Z = 0
        r_index_W = 0
        
        pos_rot_row_num = 2 if self.exists_properties_columns else 3
        axis_row_num = 3 if self.exists_properties_columns else 4

        for ti in partsIndex:

            if ti != 0 and "Position" in self.data[1].values.T[ti, pos_rot_row_num]:

                if "X" == self.data[1].values.T[ti, axis_row_num]:
                    p_index_X = ti
                if "Y" == self.data[1].values.T[ti, axis_row_num]:
                    p_index_Y = ti
                if "Z" == self.data[1].values.T[ti, axis_row_num]:
                    p_index_Z = ti

            if ti != 0 and "Rotation" in self.data[1].values.T[ti, pos_rot_row_num]:

                if "X" == self.data[1].values.T[ti, axis_row_num]:
                    r_index_X = ti
                if "Y" == self.data[1].values.T[ti, axis_row_num]:
                    r_index_Y = ti
                if "Z" == self.data[1].values.T[ti, axis_row_num]:
                    r_index_Z = ti
                if "W" == self.data[1].values.T[ti, axis_row_num]:
                    r_index_W = ti

        p_index = np.array([p_index_X, p_index_Y, p_index_Z])
        r_index = np.array([r_index_X, r_index_Y, r_index_Z, r_index_W])
        index = np.append(p_index, r_index)

        return index

    def getPartsIndex(self, unitsIndex, parts: str = None):

        # print("while getting parts index ...")

        index = np.empty(0, dtype=int)

        units_row_num = 0 if self.exists_properties_columns else 1

        if parts != None or parts != []:

            for ti in unitsIndex:
                if parts == self.data[1].values.T[ti, units_row_num]:
                    index = np.append(index, ti)

            m = np.array([0, 0, 0, 0, 0, 0, 0])
            n = 0
            for i in index:
                m[n] = i
                n = n + 1
            index = m
        else:

            index = unitsIndex

        return index

    def getUnitsIndex(self, units: str):

        # print("while getting units index ...")

        i = 0
        index = np.empty(0, dtype=int)

        if units != None:

            units_name_list = self.data[1].columns.values if self.exists_properties_columns else self.data[1].values.T[:, 0]

            for ti in units_name_list:

                if units in str(ti):
                    index = np.append(index, i)

                i += 1

        return index


class NewDataReader():

    def __init__(self, values_df: pd.DataFrame, properties_df: pd.DataFrame,
                 bone_info: dict, rigid_body_info: dict, marker_info: dict, rigid_body_marker_info: dict,
                 exists_properties_columns: bool = True):
        
        self.values_values = values_df.values
        self.properties_values = properties_df.values
        
        self.bone_info = bone_info
        self.rigid_body_info = rigid_body_info
        self.marker_info = marker_info
        self.rigid_body_marker_info = rigid_body_marker_info
        
        self.bone_flag = self.bone_info["flag"]
        self.bone_parts = self.bone_info["parts"]
        self.bone_pos_axes = self.bone_info["position"]
        self.bone_rot_axes = self.bone_info["rotation"]

        self.rigid_body_flag = self.rigid_body_info["flag"]
        self.rigid_body_parts = self.rigid_body_info["parts"]
        self.rigid_body_pos_axes = self.rigid_body_info["position"]
        self.rigid_body_rot_axes = self.rigid_body_info["rotation"]

        self.marker_flag = self.marker_info["flag"]
        self.marker_parts = self.marker_info["parts"]
        self.marker_pos_axes = self.marker_info["position"]
        self.marker_rot_axes = self.marker_info["rotation"]

        self.rigid_body_marker_flag = self.rigid_body_marker_info["flag"]
        self.rigid_body_marker_parts = self.rigid_body_marker_info["parts"]
        self.rigid_body_marker_pos_axes = self.rigid_body_marker_info["position"]
        self.rigid_body_marker_rot_axes = self.rigid_body_marker_info["rotation"]

    def _split_properties_values(self):

        self.time_series_values = self.properties_values[:, :2]  # values of the input properties in the column containing elements that named "Time" & "Frame"
        self.properties_values = self.properties_values[:, 2:]  # hereafter "properties_values" is defined as except for "time_series_values" above

        self.columns_num = self.properties_values.shape[1]

    def _search_target_column_indices(self):

        self.target_columns_indices = []

        for searching_column in range(self.columns_num):

            searching_unit = self.properties_values[0, searching_column]
            searching_part = self.properties_values[1, searching_column]
            searching_pos_rot = self.properties_values[3, searching_column]
            searching_axis = self.properties_values[4, searching_column]

            if self.bone_flag and (searching_unit == "Bone"):
                contains_target = self._judge_searching_column_exists("bone", searching_part, searching_pos_rot, searching_axis)

            elif self.rigid_body_flag and (searching_unit == "Rigid Body"):
                contains_target = self._judge_searching_column_exists("rigid_body", searching_part, searching_pos_rot, searching_axis)

            elif self.marker_info and (searching_unit == "Marker"):
                contains_target = self._judge_searching_column_exists("marker", searching_part, searching_pos_rot, searching_axis)

            elif self.marker_info and (searching_unit == "Rigid Body Marker"):
                contains_target = self._judge_searching_column_exists("rigid_body_marker", searching_part, searching_pos_rot, searching_axis)

            else:
                contains_target = False

            if contains_target:
                self.target_columns_indices.append(searching_column)

    def _judge_searching_column_exists(self, unit_name, searching_part, searching_pos_rot, searching_axis):

        if unit_name == "bone":
            if searching_part in self.bone_parts:
                if (searching_pos_rot == "Position") and (searching_axis in self.bone_pos_axes):
                    return True
                elif (searching_pos_rot == "Rotation") and (searching_axis in self.bone_rot_axes):
                    return True
                else:
                    return False
            else:
                return False

        elif unit_name == "rigid_body":
            if searching_part in self.rigid_body_parts:
                if (searching_pos_rot == "Position") and (searching_axis in self.rigid_body_pos_axes):
                    return True
                elif (searching_pos_rot == "Rotation") and (searching_axis in self.rigid_body_rot_axes):
                    return True
                else:
                    return False
            else:
                return False

        elif unit_name == "marker":
            if searching_part in self.marker_parts:
                if (searching_pos_rot == "Position") and (searching_axis in self.marker_pos_axes):
                    return True
                elif (searching_pos_rot == "Rotation") and (searching_axis in self.marker_rot_axes):
                    return True
                else:
                    return False
            else:
                return False

        elif unit_name == "rigid_body_marker":
            if searching_part in self.rigid_body_marker_parts:
                if (searching_pos_rot == "Position") and (searching_axis in self.rigid_body_marker_pos_axes):
                    return True
                elif (searching_pos_rot == "Rotation") and (searching_axis in self.rigid_body_marker_rot_axes):
                    return True
                else:
                    return False
            else:
                return False

        else:
            return False

    def _create_extracted_df(self):

        self.extracted_values_values = self.values_values[:, self.target_columns_indices]
        self.extracted_properties_values = self.properties_values[:, self.target_columns_indices]

        self.extracted_values_df = pd.DataFrame(data=self.extracted_values_values, index=None, columns=None)
        self.extracted_properties_df = pd.DataFrame(data=self.extracted_properties_values, index=None, columns=None)
    
    def _run(self):

        self._split_properties_values()
        self._search_target_column_indices()
        self._create_extracted_df()

    def get_extracted_data(self):
        
        self._run()

        return self.extracted_values_df, self.extracted_properties_df
