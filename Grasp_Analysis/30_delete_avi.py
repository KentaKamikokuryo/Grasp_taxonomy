import os
import rapidjson
from Classes.Factories import DataInfoFactory


class Manager():

    def __init__(self, experiment_name: str):

        self.experiment_name = experiment_name

    def _generate_experiment_info(self):

        self.data_info_factory = DataInfoFactory()

        self.data_info = self.data_info_factory.get_Datainfo(name=self.experiment_name)

        self.experiment_info = self.data_info.get_experiment_info()

    def _get_avi_paths(self):

        self.experiment_dir_path = self.experiment_info["experiment_dir"]
        self.takes_json = rapidjson.load(open(self.experiment_info["take_info"], 'r'))

        self.take_names = self.takes_json["take_name_list"]

        self.avi_paths = [self.experiment_dir_path + take_name + "\\result\\" + take_name + ".avi"
                          for take_name in self.take_names]

    def _delete_avi_files(self):

        for avi_path in self.avi_paths:

            print("loading the following file:", avi_path)

            os.remove(avi_path)

            print("deleted successfully the following file:", avi_path)
            print("")

        print("deleting process has been completed.")

    def run(self):

        self._generate_experiment_info()
        self._get_avi_paths()
        self._delete_avi_files()


experiment_num = 4

experiment_names = ["experiment_1", "experiment_2", "experiment_3", "experiment_4"]
experiment_name = experiment_names[experiment_num-1]

manager = Manager(experiment_name=experiment_name)
manager.run()


