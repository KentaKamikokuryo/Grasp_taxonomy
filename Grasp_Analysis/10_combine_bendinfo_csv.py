import os
import pandas as pd
import numpy as np
import rapidjson
from Classes.Info import IDataInfo
from Classes.Factories import DataInfoFactory


class Manager():

    def __init__(self, experiment_name: str):

        self.experiment_name = experiment_name
        self.take_name = "test"

        fac = DataInfoFactory()
        self.data_info = fac.get_Datainfo(name=self.experiment_name)
        self.data_info.set_data_info(name_take=self.take_name)

    def _load_json_files(self):

        self.name_file = "analysis"
        self.path_dict = self.data_info.path_dict[self.name_file]
        self.bendinfo_bend_paths = self.path_dict["bendinfo_bends"]

        self.bendinfo_bend_json_files = [rapidjson.load(open(folder+"bend.json", 'r'))
                                         for folder in self.bendinfo_bend_paths]

        self.bendinfo_bend_ids = self.path_dict["bendinfo_bend_ids"]

    def _reorganize_json_files(self):

        self.expected_grip_dicts = [file.pop("expected_grip") for file in self.bendinfo_bend_json_files]

        for i, bend_id in enumerate(self.bendinfo_bend_ids):

            self.bendinfo_bend_json_files[i].update(self.expected_grip_dicts[i])

            self.bendinfo_bend_json_files[i]["bend_id"] = int(bend_id.split("-")[0] + bend_id.split("-")[1])
            self.bendinfo_bend_json_files[i]["work_id"] = int(bend_id.split("-")[0])
            self.bendinfo_bend_json_files[i]["process"] = int(bend_id.split("-")[1])

    def _convert_json_dataframe(self):

        self.keys = self.bendinfo_bend_json_files[0].keys()
        self.bendinfo_dict = {key: [file[key] for file in self.bendinfo_bend_json_files] for key in self.keys}

        self.bendinfo_df = pd.DataFrame(self.bendinfo_dict)

    def _save_dataframe_csv(self):

        self.bendinfo_df.to_csv(self.path_dict["bendinfo_csv"], index=False)

    def run(self):

        self._load_json_files()
        self._reorganize_json_files()
        self._convert_json_dataframe()
        self._save_dataframe_csv()


#------------------------------------------------------parameters------------------------------------------------------#
experiment_num = 4
#----------------------------------------------------------------------------------------------------------------------#


experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name)
manager.run()
