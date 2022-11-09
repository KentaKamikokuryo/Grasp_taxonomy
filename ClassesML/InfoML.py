from Classes.Info import PathInfo
import os, itertools

class DataInfoML():

    def __init__(self, data_mode, hand_mode, data_level: str = None):

        self.data_mode = data_mode
        self.hand_mode = hand_mode
        self.data_level = data_level

        self.__set_csv_name()

    def __set_csv_name(self):

        if self.data_level is not None:

            self._csv_data_name = self.data_level + "_dataset_" + self.data_mode + "_kinematic_variable_" + self.hand_mode + ".csv"

        else:

            self._csv_data_name = "dataset_" + self.data_mode + "_kinematic_variable_" + self.hand_mode + ".csv"


    @property
    def csv_data_name(self):
        return self._csv_data_name

class PathInfoML():

    def __init__(self, data_name, experiment_name: str = None, test_subject_name: str = None, data_level: str = None):

        pathInfo = PathInfo()
        self._path_parent = pathInfo.path_parent_project
        self._path_parent = "C:\\Users\\Kenta Kamikokuryo\\Desktop\\AMADA_new"
        self.data_name = data_name

        self.experiment_name = experiment_name
        self.test_subject_name = test_subject_name
        self.data_level = data_level

        self._path_experiment = self._path_parent + "\\" + self.experiment_name + "\\"

        self.__set_folder()
        self.__set_folder_data()

    def __set_folder(self):

        self._folder_ML = self._path_parent + "\\ML\\"
        if not (os.path.exists(self._folder_ML)):
            os.makedirs(self._folder_ML)

        if self.test_subject_name is not None:

            if self.data_level is not None:

                self._folder_experiment = self._folder_ML + self.experiment_name + "_" + self.test_subject_name + "_test\\" + self.data_level + "\\"

            else:

                self._folder_experiment = self._folder_ML + self.experiment_name + "_" + self.test_subject_name + "_test\\"

            if not (os.path.exists(self._folder_experiment)):
                os.makedirs(self._folder_experiment)

        else:

            if self.data_level is not None:

                self._folder_experiment = self._folder_ML + self.experiment_name + "\\" + self.data_level + "\\"

            else:

                self._folder_experiment = self._folder_ML + self.experiment_name + "\\"

            if not (os.path.exists(self._folder_experiment)):
                os.makedirs(self._folder_experiment)


        self._path_figure = self._folder_experiment + "Figures\\"
        if not (os.path.exists(self._path_figure)):
            os.makedirs(self._path_figure)

        self._path_search = self._folder_experiment + "Search_Models\\"
        if not (os.path.exists(self._path_search)):
            os.makedirs(self._path_search)

        self._path_results = self._folder_experiment + "Results\\"
        if not (os.path.exists(self._path_results)):
            os.makedirs(self._path_results)

    def __set_folder_data(self):

        self._path_figure_data = self._path_figure + self.data_name + "\\"
        if not (os.path.exists(self._path_figure_data)):
            os.makedirs(self._path_figure_data)

        self._path_search_data = self._path_search + self.data_name + "\\"
        if not (os.path.exists(self._path_search_data)):
            os.makedirs(self._path_search_data)

        self._path_results_data = self._path_results + self.data_name + "\\"
        if not (os.path.exists(self._path_results_data)):
            os.makedirs(self._path_results_data)

    def set_path_model(self, model_name):

        path_search_model = self._path_search_data + model_name + "\\"
        print("Search results will be saved at and loaded from " + path_search_model)

        if not (os.path.exists(path_search_model)):
            os.makedirs(path_search_model)

        return path_search_model

    @property
    def path_experiment(self):
        return self._path_experiment

    @property
    def path_figure_data(self):
        return self._path_figure_data

    @property
    def path_search_data(self):
        return self._path_search_data

    @property
    def path_result_data(self):
        return self._path_results_data

class PathInfoKinematics():

    def __init__(self):

        pathInfo = PathInfo()
        self._path_parent = pathInfo.path_parent_project
        self._path_parent = "C:\\Users\\Kenta Kamikokuryo\\Desktop\\AMADA_new"

        self.__set_folder()

    def __set_folder(self):

        self._path_kinematics = self._path_parent + "\\Kinematics"
        if not (os.path.exists(self._path_kinematics)):
            os.makedirs(self._path_kinematics)

        self._path_corr_matrix = self._path_kinematics + "\\Corr_matrix"
        if not (os.path.exists(self._path_corr_matrix)):
            os.makedirs(self._path_corr_matrix)

    @property
    def path_kinematics(self):
        return self._path_kinematics

    @property
    def path_corr_matrix(self):
        return self._path_corr_matrix