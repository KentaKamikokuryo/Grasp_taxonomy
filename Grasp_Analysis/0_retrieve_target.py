import os
import math
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import rapidjson
from tabulate import tabulate
from Classes.Data import DataFactory, Creation_Grasp
from Classes.Factories import DataInfoFactory
from Classes.Data import DataReader, NewDataReader
from Classes.Info import IDataInfo
from Classes.Console_utilities import Color


class Manager():

    experiment_name: str
    name_file: str
    take_name: str
    speed_threshold: float

    data_info: IDataInfo
    take_names: List[str]
    data_dict: Dict

    values: pd.DataFrame
    properties: pd.DataFrame

    def __init__(self, experiment_name: str, take_name: str, running_info: Dict):

        self.experiment_name = experiment_name
        self.take_name = take_name

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.take_names = fac.get_take_names(name=self.experiment_name)

        self.running_mode = running_info["running_mode"]
        self.retrieving_mode = running_info["retrieving_mode"]
        self.batch_frame = running_info["batch_frame"]
        self.num_posture = running_info["num_posture"]

        self.machine_asset = running_info["machine_asset"]
        self.machine_name = running_info["machine_name"]
        self.moving_axis = running_info["moving_axis"]
        self.speed_threshold = running_info["speed_threshold"]

        self.position_threshold = running_info["position_threshold"]
        self.before_after_frames = running_info["before_after_frames"]
        self.before_frame_num = self.before_after_frames[0]
        self.after_frame_num = self.before_after_frames[1]
        self.duration_frame_num = self.before_frame_num + self.after_frame_num

        self.id_name = running_info["id_name"]

        self.should_cheat = running_info["should_cheat"]
        self.should_show_searching_result = running_info["should_show_searching_result"]

        self.grasp_info = running_info["grasp_info"]
        self.cheat_info = running_info["cheat_info"]

    # for "auto" running_mode
    def _initialize_auto_mode(self):

        self.experiment_dict = self.data_info.get_experiment_info()

        self.takes_json = rapidjson.load(open(self.experiment_dict["take_info"], 'r'))
        self.take_name_list = self.takes_json["take_name_list"]
        self.work_num_list = self.takes_json["work_num_list"]
        self.count_num_list = self.takes_json["count_num_list"]
        self.bend_nums_list = self.takes_json["bend_nums_list"]

        self.grasp_info = rapidjson.load(open(self.experiment_dict["grasp_info"], 'r'))

        self.cheat_info_path = self.experiment_dict["velocity_cheat_info"] if self.retrieving_mode in ["all", "average", "point"]\
            else self.experiment_dict["position_cheat_info"] if self.retrieving_mode in ["duration"]\
            else None

        self.cheat_info_df = pd.read_table(self.cheat_info_path)

        if self.retrieving_mode in ["all", "average", "point"]:
            self.speed_threshold_list = self.cheat_info_df["speed_threshold"].values
        elif self.retrieving_mode in ["duration"]:
            self.position_threshold_list = self.cheat_info_df["position_threshold"].values

        self.add_frame_ranges_list = self.cheat_info_df["add_ranges"].values
        self.remove_range_indices_list = self.cheat_info_df["remove_indices"].values

    # for "auto" running_mode
    def _initialize_running_info_for_take(self):

        self.take_name = self.take_name_list[self.take_i]

        if self.retrieving_mode in ["all", "average", "point"]:
            self.speed_threshold = self.speed_threshold_list[self.take_i]
        elif self.retrieving_mode in ["duration"]:
            self.position_threshold = self.position_threshold_list[self.take_i]

        self.cheat_info = {"add_frame_ranges": eval(self.add_frame_ranges_list[self.take_i]),
                           "remove_range_indices": eval(self.remove_range_indices_list[self.take_i])}

        self.id_name = self.take_name.split("_")[0] + "-" + self.take_name.split("_")[1] + "-" + str(self.count_num_list[self.take_i])

    def _generate_original_data_dict(self):

        self.name_file = "grasp_original"

        self.data_info.set_data_info(name_take=self.take_name)
        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

    def _generate_original_data(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.original_df = data_factory.create()

        self.original_df_value = self.original_df[0]
        self.original_df_properties = self.original_df[1]
        self.original_df_header = self.original_df[2]
        self.original_df_timeSeries = self.original_df[3]

        value = pd.concat([self.original_df_timeSeries, self.original_df_value], axis=1, ignore_index=True)
        self.df_original = value, self.original_df_properties, self.original_df_header

    def _create_velocity_data(self):

        exist_velocity_data = os.path.exists(self.data_dict["velocity"])
        if exist_velocity_data:
            print("The process that computes velocity because velocity data has already created.")
            return

        creation = Creation_Grasp(df_original=self.df_original, data_dict=self.data_dict)
        creation.get_CSV()

    def _generate_velocity_data_dict(self):

        self.name_file = "grasp_velocity"

        self.data_dict = self.data_info.get_data_dict(name_file=self.name_file)

    def _generate_velocity_data(self):

        data_factory = DataFactory(data_dict=self.data_dict, data_info=self.data_info)

        self.velocity_df = data_factory.create()

        self.velocity_df_value = self.velocity_df[0]
        self.velocity_df_properties = self.velocity_df[1]
        self.velocity_df_header = self.velocity_df[2]
        self.velocity_df_timeSeries = self.velocity_df[3]

    def _extract_machine_data(self):

        if self.machine_asset == "Rigid Body":

            bone_info = {'flag': False,
                         'parts': [],
                         'position': ["X", "Y", "Z"],
                         'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_info = {'flag': True,
                               'parts': [self.machine_name],
                               'position': [self.moving_axis],
                               'rotation': []}

            marker_info = {'flag': False,
                           'parts': [],
                           'position': ["X", "Y", "Z"],
                           'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_marker_info = {'flag': False,
                                      'parts': [],
                                      'position': ["X", "Y", "Z"],
                                      'rotation': ["W", "Z", "Y", "X"]}

        elif self.machine_asset == "Marker":

            bone_info = {'flag': False,
                         'parts': [],
                         'position': ["X", "Y", "Z"],
                         'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_info = {'flag': False,
                               'parts': [],
                               'position': ["X", "Y", "Z"],
                               'rotation': ["W", "Z", "Y", "X"]}

            marker_info = {'flag': True,
                           'parts': [self.machine_name],
                           'position': [self.moving_axis],
                           'rotation': []}

            rigid_body_marker_info = {'flag': False,
                                      'parts': [],
                                      'position': ["X", "Y", "Z"],
                                      'rotation': ["W", "Z", "Y", "X"]}

        elif self.machine_asset == "Rigid Body Marker":

            bone_info = {'flag': False,
                         'parts': [],
                         'position': ["X", "Y", "Z"],
                         'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_info = {'flag': False,
                               'parts': [],
                               'position': ["X", "Y", "Z"],
                               'rotation': ["W", "Z", "Y", "X"]}

            marker_info = {'flag': False,
                           'parts': [],
                           'position': ["X", "Y", "Z"],
                           'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_marker_info = {'flag': True,
                                      'parts': [self.machine_name],
                                      'position': [self.moving_axis],
                                      'rotation': []}

        else:

            bone_info = {'flag': False,
                         'parts': [],
                         'position': ["X", "Y", "Z"],
                         'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_info = {'flag': False,
                               'parts': [],
                               'position': ["X", "Y", "Z"],
                               'rotation': ["W", "Z", "Y", "X"]}

            marker_info = {'flag': False,
                           'parts': [],
                           'position': ["X", "Y", "Z"],
                           'rotation': ["W", "Z", "Y", "X"]}

            rigid_body_marker_info = {'flag': False,
                                      'parts': [],
                                      'position': ["X", "Y", "Z"],
                                      'rotation': ["W", "Z", "Y", "X"]}

        # parameters = {"values": pd.concat([self.velocity_df[3], self.velocity_df[0]], axis=1, ignore_index=True),
        #               "properties": self.velocity_df[1],
        #               "bone": bone_info,
        #               "rigid_body": rigid_body_info,
        #               "marker": marker_info,
        #               "exists_properties_columns": False}
        #
        # data_reader = DataReader(**parameters)
        # extracted_values_df, extracted_properties_df = data_reader.getData()

        if self.retrieving_mode in ["all", "average", "point"]:
            self.searching_values_df = self.velocity_df_value
            self.searching_properties_df = self.velocity_df_properties
        elif self.retrieving_mode in ["duration"]:
            self.searching_values_df = self.original_df_value
            self.searching_properties_df = self.original_df_properties

        parameters = {"values_df": self.searching_values_df,
                      "properties_df": self.searching_properties_df,
                      "bone_info": bone_info,
                      "rigid_body_info": rigid_body_info,
                      "marker_info": marker_info,
                      "rigid_body_marker_info": rigid_body_marker_info,
                      "exists_properties_columns": False}

        data_reader = NewDataReader(**parameters)
        extracted_values_df, extracted_properties_df = data_reader.get_extracted_data()

        self.extracted_machine_values = extracted_values_df.values.flatten()

    def _search_moving_range(self):

        self.start_indices = []
        self.end_indices = []

        self.is_moving = False

        if self.retrieving_mode in ["all", "average", "point"]:
            self.searching_threshold = self.speed_threshold
        elif self.retrieving_mode in ["duration"]:
            self.searching_threshold = self.position_threshold

        for i, value in enumerate(self.extracted_machine_values):

            if self.is_moving is False and value <= self.searching_threshold:

                self.start_indices.append(i)
                self.is_moving = True

            elif self.is_moving and value > self.searching_threshold:

                self.end_indices.append(i)
                self.is_moving = False

        self.start_end_indices_zip = zip(self.start_indices, self.end_indices)
        self.num_bends = len(list(self.start_end_indices_zip))

        self.ranges_frames = [self.end_indices[bend] - self.start_indices[bend] for bend in range(self.num_bends)]
        self.min_range_frames = min(self.ranges_frames)

        self.median_frames = [self.start_indices[bend] + int(self.ranges_frames[bend] / 2)
                              for bend in range(self.num_bends)]

    def _show_searching_result(self, fig_num: int = 1, need_figure: bool = True):

        print("searching:", self.take_name)
        print("num_bends:", self.num_bends)

        tabulate_df = pd.DataFrame(data=np.stack([self.start_indices, self.end_indices, self.ranges_frames, self.median_frames], axis=1),
                                   columns=["start frame", "end frame", "range frames", "median frame"])
        print(tabulate(tabulate_df, headers='keys', tablefmt='psql'))

        print("")

        if need_figure:

            fig = plt.figure(fig_num)
            axis = fig.add_subplot(111)

            axis.plot(self.extracted_machine_values, color="b")

            for bend in range(self.num_bends):

                start = self.start_indices[bend]
                end = self.end_indices[bend]

                axis.plot(np.arange(start, end), self.extracted_machine_values[start:end], color="r")

            plt.show()

    # for "duration" retrieving_mode
    def _show_segmentation_result(self, fig_num: int = 1, need_figure: bool = True):

        print("segmentation result:", self.take_name)
        print("num_bends:", self.num_bends)

        self.segment_start_frames = [median_frame-self.before_frame_num for median_frame in self.median_frames]
        self.segment_end_frames = [median_frame+self.after_frame_num for median_frame in self.median_frames]

        tabulate_df = pd.DataFrame(data=np.stack([self.segment_start_frames, self.segment_end_frames, self.median_frames], axis=1),
                                   columns=["start frame", "end frame", "median frame"])
        print(tabulate(tabulate_df, headers='keys', tablefmt='psql'))

        print("")

        if need_figure:

            fig = plt.figure(fig_num)
            axis = fig.add_subplot(111)

            axis.plot(self.extracted_machine_values, color="b")

            for bend in range(self.num_bends):

                start = self.segment_start_frames[bend]
                end = self.segment_end_frames[bend]

                axis.plot(np.arange(start, end), self.extracted_machine_values[start:end], color="g")

            plt.show()

    def _cheat_moving_range(self):

        self.remove_range_indices = self.cheat_info["remove_range_indices"]
        self.add_frame_ranges = np.array(self.cheat_info["add_frame_ranges"])

        self.start_indices = [self.start_indices[bend] for bend in range(self.num_bends) if bend not in self.remove_range_indices]
        self.end_indices = [self.end_indices[bend] for bend in range(self.num_bends) if bend not in self.remove_range_indices]
        self.num_bends -= len(self.remove_range_indices)
        self.ranges_frames = [self.end_indices[bend] - self.start_indices[bend] for bend in range(self.num_bends)]
        self.min_range_frames = min(self.ranges_frames)

        for frame_range in self.add_frame_ranges:
            for bend in range(self.num_bends):
                if (frame_range[0] < self.start_indices[bend]) and (frame_range[1] < self.end_indices[bend]):
                    self.start_indices.insert(bend, frame_range[0])
                    self.end_indices.insert(bend, frame_range[1])
                    self.num_bends += 1
                    break
                elif (frame_range[0] > self.start_indices[-1]) and (frame_range[1] > self.end_indices[-1]):
                    self.start_indices.append(frame_range[0])
                    self.end_indices.append(frame_range[1])
                    self.num_bends += 1
                    break

        self.ranges_frames = [self.end_indices[bend] - self.start_indices[bend] for bend in range(self.num_bends)]
        self.min_range_frames = min(self.ranges_frames)

        self.median_frames = [self.start_indices[bend] + int(self.ranges_frames[bend] / 2)
                              for bend in range(self.num_bends)]

        print("Have finished cheating process.")

    def _extract_posture_data(self):

        # parameters = {"values": self.original_df[0],
        #               "properties": self.original_df[1],
        #               "bone": self.grasp_info["bone_info"],
        #               "rigid_body": self.grasp_info["rigid_body_info"],
        #               "marker": self.grasp_info["marker_info"],
        #               "exists_properties_columns": False}
        #
        # data_reader = DataReader(**parameters)
        # extracted_values_df, extracted_properties_df = data_reader.getData()

        parameters = {"values_df": self.original_df_value,
                      "properties_df": self.original_df_properties,
                      "bone_info": self.grasp_info["bone_info"],
                      "rigid_body_info": self.grasp_info["rigid_body_info"],
                      "marker_info": self.grasp_info["marker_info"],
                      "rigid_body_marker_info": self.grasp_info["rigid_body_marker_info"],
                      "exists_properties_columns": False}

        data_reader = NewDataReader(**parameters)
        extracted_values_df, extracted_properties_df = data_reader.get_extracted_data()
        self.extracted_posture_values = extracted_values_df.values
        self.extracted_properties_values = extracted_properties_df.values

    def _replace_posture_data_nan(self):

        for column in range(self.extracted_posture_values.shape[1]):
            self.extracted_posture_values[np.isnan(self.extracted_posture_values)[:, column].ravel(), column]\
                = np.nanmean(self.extracted_posture_values[:, column].ravel())

    # for "all" or "average" retrieving_mode
    def _average_posture_data(self):

        self.postures_bends = []  # posture: a(an averaged) posture at one frame

        for bend in range(self.num_bends):

            start = self.start_indices[bend]
            end = self.end_indices[bend]

            self.postures_bends.append(self.extracted_posture_values[start:end])

        self.mean_bends = np.array([np.mean(postures_bend, axis=0) for postures_bend in self.postures_bends])

    def _check_all_mode_executable(self):

        self.min_iterations = int(self.min_range_frames / self.batch_frame)

        self.can_execute_all_mode = True if self.min_iterations >= self.num_posture else False

        if self.retrieving_mode == "all" and self.can_execute_all_mode is False:
            print(f"{Color.YELLOW}Can't run the \"all\" mode because the \"batch_frame\" or the \"num_posture\" is too big."
                  f"\nSwitch the \"average\" mode.{Color.RESET}")
            self.retrieving_mode = "average"

    # for "all" retrieving_mode
    def _divide_batches(self):

        self.postures_batches_bends = [[self.postures_bends[bend][iteration*self.batch_frame:(iteration+1)*self.batch_frame, :]
                                        for iteration in range(int(len(self.postures_bends[bend])/self.batch_frame))]
                                       for bend in range(self.num_bends)]

    # for "all" retrieving_mode
    def _average_batches(self):

        self.mean_batches_bends = [np.array([np.mean(self.postures_batches_bends[bend][iteration], axis=0)
                                             for iteration in range(len(self.postures_batches_bends[bend]))])
                                   for bend in range(self.num_bends)]

        self.loss_batches_bends = [np.array([mean_squared_error(self.mean_bends[bend, :], self.mean_batches_bends[bend][iteration, :])
                                             for iteration in range(len(self.postures_batches_bends[bend]))])
                                   for bend in range(self.num_bends)]
        self.indices_batches_bends = [np.argsort(self.loss_batches_bends[bend]) for bend in range(self.num_bends)]

        self.limited_mean_batches_bends = [self.mean_batches_bends[bend][self.indices_batches_bends[bend][:self.num_posture], :]
                                           for bend in range(self.num_bends)]

    # for "point" retrieving_mode
    def _extract_median_posture_data(self):

        self.posture_bends = []  # posture: a(an averaged) posture at one frame

        for bend in range(self.num_bends):

            median = self.median_frames[bend]

            self.posture_bends.append(self.extracted_posture_values[median])

        self.posture_bends = np.array(self.posture_bends)

    # for "duration" retrieving_mode
    def _segment_motion_data(self):

        self.motion_bends = []  # motion: time-series data that consists of some frames

        for bend in range(self.num_bends):

            median = self.median_frames[bend]
            start = median - self.before_frame_num
            end = median + self.after_frame_num

            self.motion_bends.append(self.extracted_posture_values[start:end])

        self.motion_bends = np.array(self.motion_bends).reshape([-1, self.extracted_posture_values.shape[1]])

    # for "average" retrieving_mode
    def _concatenate_original_averaged_data(self):

        self.labels_values = np.array([[self.id_name+"_"+str(bend)+"_AVE", "bend_"+str(bend)]
                                       for bend in range(self.num_bends)])

        revised_properties_values_left = np.empty([5, 2], dtype=object)
        revised_properties_values_left[:, :] = np.nan
        revised_properties_values_left[-1, 0] = "Id"
        revised_properties_values_left[-1, 1] = "Name"
        revised_properties_values_right = self.extracted_properties_values
        revised_properties_values = np.concatenate([revised_properties_values_left, revised_properties_values_right], axis=1)

        all_values_averaged = np.concatenate([self.labels_values, self.mean_bends], axis=1)
        except_header_averaged = np.concatenate([revised_properties_values, all_values_averaged], axis=0)

        difference_header_column = except_header_averaged.shape[1] - len(self.original_df_header.columns)
        wider_values = "properties" if difference_header_column > 0 else "header" if difference_header_column < 0 else "equal"
        difference_header_column = abs(difference_header_column)

        header_values = self.original_df_header.values

        if wider_values == "properties":

            nan_ndarray = np.empty([2, difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            header_values = np.concatenate([header_values, nan_ndarray], axis=1)

        elif wider_values == "header":

            nan_ndarray = np.empty([self.num_bends+revised_properties_values.shape[0], difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            except_header_averaged = np.concatenate([except_header_averaged, nan_ndarray], axis=1)

        original_values_averaged = np.concatenate([header_values, except_header_averaged], axis=0)

        self.df_original_averaged = pd.DataFrame(original_values_averaged)

    # for "all" retrieving_mode
    def _concatenate_original_all_data(self):

        self.labels_values = np.array([[self.id_name+"_"+str(bend)+"_"+str(iteration)+"_ALL", "bend_" + str(bend)]
                                       for bend in range(self.num_bends)
                                       for iteration in range(self.num_posture)])

        revised_properties_values_left = np.empty((5, 2), dtype=object)
        revised_properties_values_left[:, :] = np.nan
        revised_properties_values_left[-1, 0] = "Id"
        revised_properties_values_left[-1, 1] = "Name"
        revised_properties_values_right = self.extracted_properties_values
        revised_properties_values = np.concatenate([revised_properties_values_left, revised_properties_values_right], axis=1)

        all_values_batches = np.concatenate([self.labels_values,
                                             np.array(self.limited_mean_batches_bends).reshape([self.num_posture*self.num_bends, -1])], axis=1)
        except_header_batches = np.concatenate([revised_properties_values, all_values_batches], axis=0)

        difference_header_column = except_header_batches.shape[1] - len(self.original_df_header.columns)
        wider_values = "properties" if difference_header_column > 0 else "header" if difference_header_column < 0 else "equal"
        difference_header_column = abs(difference_header_column)

        header_values = self.original_df_header.values

        if wider_values == "properties":

            nan_ndarray = np.empty([2, difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            header_values = np.concatenate([header_values, nan_ndarray], axis=1)

        elif wider_values == "header":

            nan_ndarray = np.empty([except_header_batches.shape[0], difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            except_header_batches = np.concatenate([except_header_batches, nan_ndarray], axis=1)

        original_values_batches = np.concatenate([header_values, except_header_batches], axis=0)

        self.df_original_batches = pd.DataFrame(original_values_batches)

    # for "point" retrieving_mode
    def _concatenate_original_point_data(self):

        self.labels_values = np.array([[self.id_name + "_" + str(bend) + "_PT", "bend_" + str(bend)]
                                       for bend in range(self.num_bends)])

        revised_properties_values_left = np.empty([5, 2], dtype=object)
        revised_properties_values_left[:, :] = np.nan
        revised_properties_values_left[-1, 0] = "Id"
        revised_properties_values_left[-1, 1] = "Name"
        revised_properties_values_right = self.extracted_properties_values
        revised_properties_values = np.concatenate([revised_properties_values_left, revised_properties_values_right], axis=1)

        all_values_point = np.concatenate([self.labels_values, self.posture_bends], axis=1)
        except_header_point = np.concatenate([revised_properties_values, all_values_point], axis=0)

        difference_header_column = except_header_point.shape[1] - len(self.original_df_header.columns)
        wider_values = "properties" if difference_header_column > 0 else "header" if difference_header_column < 0 else "equal"
        difference_header_column = abs(difference_header_column)

        header_values = self.original_df_header.values

        if wider_values == "properties":

            nan_ndarray = np.empty([2, difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            header_values = np.concatenate([header_values, nan_ndarray], axis=1)

        elif wider_values == "header":

            nan_ndarray = np.empty([self.num_bends+revised_properties_values.shape[0], difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            except_header_point = np.concatenate([except_header_point, nan_ndarray], axis=1)

        original_values_point = np.concatenate([header_values, except_header_point], axis=0)

        self.df_original_point = pd.DataFrame(original_values_point)

    # for "duration" retrieving_mode
    def _concatenate_original_duration_data(self):

        self.labels_values = np.array([[self.id_name + "_" + str(bend) + "_TS", "bend_" + str(bend), frame]
                                       for bend in range(self.num_bends)
                                       for frame in range(self.duration_frame_num)])

        revised_properties_values_left = np.empty([5, 3], dtype=object)
        revised_properties_values_left[:, :] = np.nan
        revised_properties_values_left[-1, 0] = "Id"
        revised_properties_values_left[-1, 1] = "Name"
        revised_properties_values_left[-1, 2] = "Frame"
        revised_properties_values_right = self.extracted_properties_values
        revised_properties_values = np.concatenate([revised_properties_values_left, revised_properties_values_right], axis=1)

        all_values_duration = np.concatenate([self.labels_values, self.motion_bends], axis=1)
        except_header_duration = np.concatenate([revised_properties_values, all_values_duration], axis=0)

        difference_header_column = except_header_duration.shape[1] - len(self.original_df_header.columns)
        wider_values = "properties" if difference_header_column > 0 else "header" if difference_header_column < 0 else "equal"
        difference_header_column = abs(difference_header_column)

        header_values = self.original_df_header.values

        if wider_values == "properties":

            nan_ndarray = np.empty([2, difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            header_values = np.concatenate([header_values, nan_ndarray], axis=1)

        elif wider_values == "header":

            nan_ndarray = np.empty([self.num_bends*self.duration_frame_num+revised_properties_values.shape[0], difference_header_column], dtype=object)
            nan_ndarray[:, :] = np.nan

            except_header_duration = np.concatenate([except_header_duration, nan_ndarray], axis=1)

        original_values_duration = np.concatenate([header_values, except_header_duration], axis=0)

        self.df_original_duration = pd.DataFrame(original_values_duration)

    # for "average" retrieving_mode
    def _save_averaged_grasp_data(self):

        self.path_averaged = self.data_dict["average"]

        self.df_original_averaged.to_csv(self.path_averaged, header=False, index=False)

    # for "all" retrieving_mode
    def _save_all_grasp_data(self):

        self.path_all = self.data_dict["all"]

        self.df_original_batches.to_csv(self.path_all, header=False, index=False)

    # for "point" retrieving_mode
    def _save_point_grasp_data(self):

        self.path_point = self.data_dict["point"]

        self.df_original_point.to_csv(self.path_point, header=False, index=False)

    # for "duration" retrieving_mode
    def _save_duration_grasp_data(self):

        self.path_duration = self.data_dict["duration"]

        self.df_original_duration.to_csv(self.path_duration, header=False, index=False)

    def _run_manual_mode(self):

        self._generate_original_data_dict()
        self._generate_original_data()

        self._create_velocity_data()
        self._generate_velocity_data_dict()
        self._generate_velocity_data()
        self._extract_machine_data()
        self._search_moving_range()

        if self.should_show_searching_result:
            self._show_searching_result()

        if self.should_cheat:
            self._cheat_moving_range()
            if self.should_show_searching_result:
                self._show_searching_result(fig_num=2)

        self._extract_posture_data()
        self._replace_posture_data_nan()
        self._average_posture_data()
        self._check_all_mode_executable()

        if self.retrieving_mode == "average":
            self._concatenate_original_averaged_data()
            self._save_averaged_grasp_data()

        elif self.retrieving_mode == "all":
            self._divide_batches()
            self._average_batches()
            self._concatenate_original_all_data()
            self._save_all_grasp_data()

        elif self.retrieving_mode == "point":
            self._extract_median_posture_data()
            self._concatenate_original_point_data()
            self._save_point_grasp_data()

        elif self.retrieving_mode == "duration":
            self._segment_motion_data()
            self._concatenate_original_duration_data()
            self._save_duration_grasp_data()
            if self.should_show_searching_result:
                self._show_segmentation_result(fig_num=3)

    def _run_auto_mode(self):

        self._initialize_auto_mode()

        for take_i in range(len(self.take_name_list)):

            self.take_i = take_i

            self._initialize_running_info_for_take()

            self._generate_original_data_dict()
            self._generate_original_data()

            self._create_velocity_data()
            self._generate_velocity_data_dict()
            self._generate_velocity_data()
            self._extract_machine_data()
            self._search_moving_range()

            if self.should_show_searching_result:
                self._show_searching_result()

            if self.should_cheat:
                self._cheat_moving_range()
                if self.should_show_searching_result:
                    self._show_searching_result(fig_num=2, need_figure=False)

            self._extract_posture_data()
            self._replace_posture_data_nan()
            self._average_posture_data()
            self._check_all_mode_executable()

            if self.retrieving_mode == "average":
                self._concatenate_original_averaged_data()
                self._save_averaged_grasp_data()

            elif self.retrieving_mode == "all":
                self._divide_batches()
                self._average_batches()
                self._concatenate_original_all_data()
                self._save_all_grasp_data()

            elif self.retrieving_mode == "point":
                self._extract_median_posture_data()
                self._concatenate_original_point_data()
                self._save_point_grasp_data()

            elif self.retrieving_mode == "duration":
                self._segment_motion_data()
                self._concatenate_original_duration_data()
                self._save_duration_grasp_data()
                if self.should_show_searching_result:
                    self._show_segmentation_result(fig_num=3, need_figure=False)

    def run(self):

        if self.running_mode == "manual":
            self._run_manual_mode()

        elif self.running_mode == "auto":
            self._run_auto_mode()


# -----------------------------------------------------parameters----------------------------------------------------- #
# experiment_num = 2
experiment_num = 5
take_name = "tera_2_202206241711"
take_name = "haga_4_202208251241"
# take_name = "test"

grasp_info = {"bone_info": {'flag': True,
                            'parts': ["Skeleton 001_LFArm", "Skeleton 001_LHand",

                                      "Skeleton 001_RFArm", "Skeleton 001_RHand"],
                            'position': ["X", "Y", "Z"],
                            'rotation': []},

              "rigid_body_info": {'flag': True,
                                  'parts': ["LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
                                            "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
                                            "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
                                            "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
                                            "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",

                                            "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
                                            "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
                                            "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
                                            "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
                                            "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4"],
                                  'position': ["X", "Y", "Z"],
                                  'rotation': []},

              "marker_info": {'flag': True,
                              'parts': ["Skeleton 001:LWristOut", "Skeleton 001:LWristIn", "Skeleton 001:LHandOut",

                                        "Skeleton 001:RWristOut", "Skeleton 001:RWristIn", "Skeleton 001:RHandOut"],
                              'position': ["X", "Y", "Z"],
                              'rotation': []},

              "rigid_body_marker_info": {'flag': False,
                                         'parts': [],
                                         'position': ["X", "Y", "Z"],
                                         'rotation': []}}


cheat_info = {"add_frame_ranges": [[3000, 3100]],
              "remove_range_indices": [0, 2]}

running_info = {# "running_mode": "manual",
                "running_mode": "auto",
                # "retrieving_mode": "average",
                # "retrieving_mode": "all",
                # "retrieving_mode": "point",
                "retrieving_mode": "duration",
                "batch_frame": 3,  # this variable is only used when retrieving_mode is "all"
                "num_posture": 5,  # this variable is only used when retrieving_mode is "all"
                "machine_asset": "Rigid Body",
                # "machine_asset": "Marker",
                "machine_name": "RigidBody 003",
                # "machine_name": "Unlabeled 5756",
                "moving_axis": "Y",
                "speed_threshold": -0.07,
                "position_threshold": 1.26,  # this variable is only used when retrieving_mode is "duration"
                "before_after_frames": [90, 90],  # this variable is only used when retrieving_mode is "duration"
                "id_name": take_name.split("_")[0]+"-"+take_name.split("_")[1]+"-2",
                # "id_name": take_name.split("-")[1]+take_name.split("-")[2]+take_name.split("-")[3],
                "should_cheat": True,
                "should_show_searching_result": True,
                "grasp_info": grasp_info,
                "cheat_info": cheat_info}
# -------------------------------------------------------------------------------------------------------------------- #


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, take_name=take_name, running_info=running_info)
manager.run()
