import os
import numpy as np
import pandas as pd
from Classes.Console_utilities import Color, InputFunctions
from Classes.Info import IDataInfo
from Classes.Factories import DataInfoFactory
from Classes.Leveled_binary_dendrogram import LeveledBinaryDendrogram


class Manager():

    data_info: IDataInfo

    def __init__(self, experiment_name: str, running_info: dict):

        self.experiment_name = experiment_name

        self.running_mode = running_info["running_mode"]
        self.from_bendinfo = running_info["from_bendinfo"]
        self.bendinfo_label_key_list = running_info["bendinfo_label_key_list"]
        self.labels_list = running_info["labels_list"]
        self.should_append_bendinfo_labels = running_info["should_append_bendinfo_labels"]
        self.append_key_list = running_info["append_key_list"]

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.experiment_dict = self.data_info.get_experiment_info()

        self.dendrogram_csv_path = self.experiment_dict["dendrogram_csv"]
        self.bendinfo_path = self.experiment_dict["bend_info"]

    def _check_file_exists(self):

        if os.path.exists(self.dendrogram_csv_path):
            print("Exists:", self.dendrogram_csv_path)

            if self.running_mode == "new":
                print("running_mode: \"new\"")
                print("The above file will be overwritten...")

            elif self.running_mode == "edit":
                print("running_mode: \"edit\"")
                print("The above file will be loaded and be edited...")

        else:

            if self.running_mode == "new":
                print("running_mode: \"new\"")
                print("A new file will be created to:", self.dendrogram_csv_path)

            elif self.running_mode == "edit":
                print(f"{Color.YELLOW}\"running_mode\" will be switched \"edit\" to \"new\" because the specified path dose not exist.{Color.RESET}")
                self.running_mode = "new"
                print("running_mode: \"new\"")
                print("A new file will be created to:", self.dendrogram_csv_path)

    def _load_bendinfo(self):

        self.bendinfo_df = pd.read_csv(self.bendinfo_path, sep=",")
        print("Loaded bendinfo:", self.bendinfo_path)

        self.bendinfo_label_df = self.bendinfo_df[self.bendinfo_label_key_list]

    def _initialize_new_mode(self):

        unique_label_list = np.unique(self.bendinfo_label_df.values) if self.from_bendinfo else self.labels_list

        self.leveled_binary_dendrogram = LeveledBinaryDendrogram(list(unique_label_list))

    def _initialize_edit_mode(self):

        self.leveled_binary_dendrogram = LeveledBinaryDendrogram()
        self.leveled_binary_dendrogram.load_file(table_path=self.dendrogram_csv_path)

        print("Loaded dendrogram csv:", self.dendrogram_csv_path)

    def _write_table(self):

        print("Start to edit the dendrogram table.")

        while True:

            self.leveled_binary_dendrogram.show_table()

            if self.leveled_binary_dendrogram.defined_level == 0:
                print("(a) -> append a level, (q) -> quit")
                acceptable_keys = ["a", "q"]
                input_key = input("Enter one of the above keys: ")

            elif self.leveled_binary_dendrogram.defined_level == self.leveled_binary_dendrogram.depth:
                print("(r) -> remove levels, (c) -> clear all levels, (q) -> quit")
                acceptable_keys = ["r", "c", "q"]
                input_key = input("Enter one of the above keys: ")

            else:
                print("(a) -> append a level, (r) -> remove levels, (c) -> clear all levels, (q) -> quit")
                acceptable_keys = ["a", "r", "c", "q"]
                input_key = input("Enter one of the above keys: ")

            if InputFunctions.check_validity(input_str=input_key, acceptable_str_list=acceptable_keys):
                if input_key == "a":
                    self._append_level()
                elif input_key == "r":
                    self._remove_levels()
                elif input_key == "c":
                    self._clear_levels()
                elif input_key == "q":
                    print("Quit the process of editing the dendrogram table.\n")
                    break

            else:
                print(f"{Color.RED}Input key is invalid!!{Color.RESET}")
                print("Enter again.\n")

    def _append_level(self):

        print("Append a new level.")

        first_index = int(input("Enter the first integrated index: "))
        second_index = int(input("Enter the second integrated index: "))
        self.leveled_binary_dendrogram.append_level(cluster_index1=first_index, cluster_index2=second_index)

        print("Completed the integration.\n")

    def _remove_levels(self):

        print("Remove some levels.")

        kept_level = int(input("Enter the kept level: "))
        self.leveled_binary_dendrogram.remove_levels(kept_level=kept_level)

        print("Completed the removing.\n")

    def _clear_levels(self):

        print("Clear the all levels.")

        self.leveled_binary_dendrogram.clear_levels()

        print("Completed the clear.\n")

    def _save_table(self):

        print("The completed dendrogram table is as follows:")
        self.leveled_binary_dendrogram.show_table()

        self.leveled_binary_dendrogram.save_file(table_path=self.dendrogram_csv_path)

    def _append_bendinfo_labels(self):

        print("Append new label columns to the bendinfo.")
        self.leveled_binary_dendrogram.show_table()

        new_labels_level = int(input("Select a level from the table above"))
        base_labels = list(self.bendinfo_label_df.values.flatten())

        self.new_labels = self.leveled_binary_dendrogram.convert_labels(labels=base_labels, level=new_labels_level)
        self.new_labels_values = np.array(self.new_labels).reshape([-1, len(self.bendinfo_label_key_list)])

        self.append_bendinfo_df = pd.DataFrame(data=self.new_labels_values,
                                               columns=self.append_key_list,
                                               index=self.bendinfo_df.index.values)

        self.new_bendinfo_df = pd.concat([self.bendinfo_df, self.append_bendinfo_df], axis=1)

        self.new_bendinfo_df.to_csv(self.bendinfo_path, index=False)
        print("Saved the new bendinfo to:", self.bendinfo_path)

    def run(self):

        self._check_file_exists()
        self._load_bendinfo()

        if self.running_mode == "new":
            self._initialize_new_mode()
        elif self.running_mode == "edit":
            self._initialize_edit_mode()

        self._write_table()

        if self.leveled_binary_dendrogram.defined_level != self.leveled_binary_dendrogram.depth:
            print("Editing process has not completed.")
            print("Exit the program.")
            return

        self._save_table()

        if self.should_append_bendinfo_labels:
            self._append_bendinfo_labels()

        print("Exit the program.")
        return


# -----------------------------------------------------parameters----------------------------------------------------- #
experiment_num = 5

running_info = {# "running_mode": "new",
                "running_mode": "edit",
                "from_bendinfo": True,  # this variable is only used when running_mode is "new"
                "bendinfo_label_key_list": ["left_hand_posture", "right_hand_posture"],
                "labels_list": list(np.arange(0, 16)) + [99],  # this variable is only used when from_bendinfo is "True"
                "should_append_bendinfo_labels": True,
                "append_key_list": ["dendrogram_left_hand_posture", "dendrogram_right_hand_posture"]}
# -------------------------------------------------------------------------------------------------------------------- #


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name, running_info=running_info)
manager.run()
