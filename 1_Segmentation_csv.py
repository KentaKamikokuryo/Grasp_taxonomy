import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Classes.Info import PathInfo, DataInfo1, DataInfo2
from Classes.Plot import Latent_space
from Classes.Classifier import ClassifierWardOrder_N
from Classes.Classifier import ClassifierWardOrder_D
from Classes.Factories import DataInfoFactory


class Manager():

    def __init__(self, segmentation_info, save_info, display_info):

        self.experiment_name = segmentation_info["experiment_name"]
        self.name_file = segmentation_info["name_file"]
        self.take_name = segmentation_info["take_name"]
        self.take_fps = segmentation_info["take_fps"]
        self.segmentation_fps = segmentation_info["segmentation_fps"]
        self.model_name = segmentation_info["model_name"]
        self.segmentation_type = segmentation_info["segmentation_type"]
        self.N = segmentation_info["N"]
        self.D = segmentation_info["D"]
        self.indices_segments = segmentation_info["indices_segments"]
        self.start_n_indices_segments = segmentation_info["start_n_indices_segments"]

        self.save_plot = save_info["plot"]
        self.n_segments = save_info["n_segments"]

        self.display_mode = display_info["mode"]

        self.data_info = segmentation_info["data_info"]
        self.data_info.set_data_info(name_take=take_name)
        self.dict_data_path = self.data_info.get_data_dict(name_file=self.name_file)

        self.path_csv = self.dict_data_path["data"]
        self.path_result = self.dict_data_path["result"]
        self.path_indices = self.dict_data_path["indices"]

    def split_csv(self):

        self.df = DataFrame()
        self.df_values, self.df_properties = self.df.load_csv(path_csv=self.path_csv)

        list_unit = list(self.df_values.columns.values)
        list_unit = [l for l in list_unit if 'Rigid Body Marker' not in l]

        list_df_bone = [l for l in list_unit if 'Bone' in l]
        list_df_rigid = [l for l in list_unit if 'Rigid Body' in l]
        list_df_marker = [l for l in list_unit if 'Marker' in l]

        print("list_df_bone:", list_df_bone)
        print("list_df_rigid:", list_df_rigid)
        print("list_df_marker:", list_df_marker)

        df_bone = self.df_values[list_df_bone]
        df_rigid = self.df_values[list_df_rigid]
        df_marker = self.df_values[list_df_marker]

        print("df_bone:")
        print(df_bone.head())
        print("df_rigid:")
        print(df_rigid.head())
        print("df_marker:")
        print(df_marker.head())

        index_df_bone_position = [i for i in range(len(list_df_bone)) if i % 7 == 4 or i % 7 == 5 or i % 7 == 6]
        print("index_df_bone_position:", index_df_bone_position)

        df_bone_position = df_bone.iloc[:, index_df_bone_position]
        print("df_bone_position:")
        print(df_bone_position.head())

        values_bone_position = df_bone_position.values
        print("values_bone_position:")
        print(values_bone_position)

        self.standardized_data = StandardScaler().fit_transform(values_bone_position)
        print("standardized_data:")
        print(self.standardized_data)

        list_missing_frame = np.unique(np.where(np.isnan(self.standardized_data))[0]).tolist()
        list_missing_location = list(zip(*np.where(np.isnan(self.standardized_data))))
        print("list_missing_location:", list_missing_location)
        print("list_missing_frame:", list_missing_frame)

        self.standardized_data = np.delete(self.standardized_data, list_missing_frame, axis=0)

        self.standardized_data = self.standardized_data[[f for f in range(len(self.standardized_data))
                                                         if f % (self.take_fps / self.segmentation_fps) == 0], :]

        print("Have installed data!")

    def reduce_dimensionally(self):

        if self.model_name == "PCA":

            pca = PCA()
            pca.n_components = 2
            pca_fit = pca.fit(self.standardized_data)
            self.reduced_data = pca_fit.transform(self.standardized_data)

        else:
            return

        figure_name = self.take_name + "_" + self.model_name
        Latent_space.plot_latent_space(X_train=self.reduced_data,
                                       is_save=self.save_plot,
                                       path_save=self.path_result+figure_name+".png")

        print("Have reduced dimensionally!")

    def segmentation_df(self):

        if self.segmentation_type == "N":

            self.cwo = ClassifierWardOrder_N(n_clusters=self.N, list_save_n=self.n_segments, display_mode=self.display_mode)

            figure_name = self.take_name + "_segmentation_" + self.segmentation_type + "=" + str(self.N)

        elif self.segmentation_type == "D":

            self.cwo = ClassifierWardOrder_D(threshold=self.D, list_save_n=self.n_segments)

            figure_name = self.take_name + "_segmentation_" + self.segmentation_type + "=" + str(self.D)

        else:
            return

        if (self.indices_segments is None) and (self.start_n_indices_segments is not None):

            self.indices_segments = self.cwo.load_indices(path_save=self.path_indices, take_name=self.take_name, n_indices_clusters=self.start_n_indices_segments)

        print("Start to fit:", self.take_name)
        self.cwo.fit(X=self.reduced_data, indices_clusters=self.indices_segments)

        self.labels = self.cwo.get_labels()
        # print("labels:", self.labels)

        self.cwo.save_indices(path_save=self.path_indices, take_name=self.take_name)

        Latent_space.plot_latent_space(X_train=self.reduced_data, y_train=self.labels,
                                       is_save=self.save_plot, path_save=self.path_result+figure_name+".png")

class DataFrame():

    def __init__(self):
        pass

    def load_csv(self, path_csv):

        self.values = pd.read_csv(path_csv, sep=",", skiprows=[0, 1, 3, 4, 5, 6])
        self.properties = pd.read_csv(path_csv, sep=",", header=1, nrows=4)

        print("Loaded following values:")
        print(self.values.head())
        print("Loaded following properties:")
        print(self.properties)

        return self.values, self.properties

# take_names= [take_names[5]]

experiment_names = ["experiment_1", "experiment_2", "experiment_3"]
experiment_name = experiment_names[2]
fac = DataInfoFactory()
dataInfo = fac.get_Datainfo(name=experiment_name)
take_names = fac.get_take_names(name=experiment_name)

name_file = "original"

for take_name in take_names:

    segmentation_info = {"experiment_name": experiment_name,
                         "data_info": dataInfo,
                         "name_file": name_file,
                         "take_name": take_name,
                         "take_fps": 90,
                         "segmentation_fps": 90,
                         "model_name": "PCA",  # "PCA" only for now, method of dimensionality reduction
                         "segmentation_type": "N",  # "N" or "D", "N": do clustering to defined number of segments, "D": do clustering until Euclidian distance reaches defined value
                         "N": 10,  # this is used when segmentation_type=N, how many segments do be clustered
                         "D": 1.0,  # this is used when segmentation_type=D, threshold of Euclidean distance
                         "indices_segments": None,  # for writing by list directly
                         "start_n_indices_segments": None}  # define None in the case of that you haven't had indices_segments npy file yet, this will ignore when indices_segments is not None

    save_info = {"plot": True,  # whether you want to save latent space
                 "n_segments": [10, 12, 50, 100, 200, 300, 500]}  # numbers of segments of indices_segments you want to save

    display_info = {"mode": "tqdm"}  # "normal" or "tqdm", what displays while fittig, "normal": display min_distance, index_min_distance, indices_clusters, n_clusters_current every clustering, "tqdm": display only progress bar

    manager = Manager(segmentation_info=segmentation_info, save_info=save_info, display_info=display_info)
    manager.split_csv()
    manager.reduce_dimensionally()
    manager.segmentation_df()
