import numpy as np
import pandas as pd
from tabulate import tabulate
from Classes.Console_utilities import Color


class LeveledBinaryDendrogram():

    def __init__(self, label_list: list = None):

        self._label_list = label_list if label_list is not None else [0]
        self._num_label = len(self.label_list)
        self._depth = self.num_label - 1
        self._defined_level = 0

        self.dendrogram_df_values = np.empty([self.num_label-1, self.num_label])
        self.dendrogram_df_values[:, :] = np.nan
        self.dendrogram_df_values = np.concatenate([np.arange(self.num_label).reshape([1, -1]), self.dendrogram_df_values], axis=0)
        self.dendrogram_df_columns = self.label_list
        self.dendrogram_df_index = np.arange(self.num_label)

        self._dendrogram_df = pd.DataFrame(data=self.dendrogram_df_values,
                                           columns=self.dendrogram_df_columns,
                                           index=self.dendrogram_df_index)

    # region properties
    @property
    def label_list(self):
        return self._label_list

    @property
    def num_label(self):
        return self._num_label

    @property
    def depth(self):
        return self._depth

    @property
    def defined_level(self):
        return self._defined_level

    @property
    def dendrogram_df(self):
        return self._dendrogram_df
    # endregion

    def append_level(self, cluster_index1: int, cluster_index2: int):

        if self.defined_level >= self.depth:
            print(f"{Color.RED}This dendrogram has already defined completely!!{Color.RESET}")
            return

        if (cluster_index1 < 0) or (cluster_index1 > self.depth-self.defined_level)\
                or (cluster_index2 < 0) or (cluster_index2 > self.depth-self.defined_level):
            print(f"{Color.RED}Input value is invalid!!{Color.RESET}")
            return

        if cluster_index1 == cluster_index2:
            print(f"{Color.RED}Don't input the same values!!{Color.RESET}")
            return

        if cluster_index1 < cluster_index2:
            smaller_index = cluster_index1
            greater_index = cluster_index2
        else:
            smaller_index = cluster_index2
            greater_index = cluster_index1

        before_level_indices = self.dendrogram_df.iloc[self.defined_level, :]
        append_level_indices = np.empty([self.num_label])

        for i, index in enumerate(before_level_indices):

            if index < greater_index:
                append_level_indices[i] = index

            elif index == greater_index:
                append_level_indices[i] = smaller_index

            elif index > greater_index:
                append_level_indices[i] = index - 1

        self.dendrogram_df.iloc[self.defined_level+1, :] = append_level_indices
        self._defined_level += 1

    def remove_levels(self, kept_level: int):

        if kept_level > self.defined_level:
            print(f"{Color.RED}This level has not defined yet!!{Color.RESET}")
            return

        if kept_level < 0:
            print(f"{Color.RED}Input value is invalid!!{Color.RESET}")
            return

        self.dendrogram_df.iloc[kept_level+1:, :] = np.nan
        print("Removed the following levels:", np.arange(kept_level+1, self.defined_level+1))
        self._defined_level = kept_level

    def clear_levels(self):

        self.dendrogram_df.iloc[1:, :] = np.nan
        print("Cleared all levels")
        self._defined_level = 0

    def load_file(self, table_path: str, sep: str = ","):

        self._dendrogram_df = pd.read_csv(table_path, sep=sep, index_col=0)

        self._label_list = [int(label) for label in list(self.dendrogram_df.columns.values)]
        self._num_label = len(self.label_list)
        self._depth = self.num_label - 1
        self._defined_level = self.depth

        self.dendrogram_df_values = self.dendrogram_df.values
        self.dendrogram_df_columns = self.dendrogram_df.columns
        self.dendrogram_df_index = self.dendrogram_df.index

    def save_file(self, table_path: str, sep: str = ","):

        if self.defined_level != self.depth:
            print(f"{Color.RED}This dendrogram has not completed yet!!{Color.RESET}")
            return

        print("Saving the dendrogram table file...")
        self.dendrogram_df.to_csv(table_path, sep=sep, header=True, index=True, index_label="Level")
        print("Have finished saving to", table_path)

    def convert_labels(self, labels: list, level: int = 0):

        if not(set(labels) <= set(self.label_list)):
            print(f"{Color.RED}Input value is invalid!!{Color.RESET}")
            return labels

        if level > self.defined_level:
            print(f"{Color.RED}Input level has not defined yet!!{Color.RESET}")
            return labels

        before_keys_list = self.label_list
        after_values_list = self.dendrogram_df.values[level].tolist()
        conversion_dict = dict(zip(before_keys_list, after_values_list))

        new_labels = [conversion_dict[label] for label in labels]

        return new_labels

    def show_table(self):

        print("num_label:", self.num_label)
        print("depth:", self.depth)
        print("defined_level:", self.defined_level)

        space_column = np.empty([self.num_label, 1], dtype=object)
        space_column[:, :] = ""
        tabulate_data = np.concatenate([self.dendrogram_df.index.values.reshape([-1, 1]),
                                        space_column,
                                        self.dendrogram_df.values],
                                       axis=1)
        tabulate_columns = ["Level"] + [""] + list(self.dendrogram_df.columns.values)

        tabulate_df = pd.DataFrame(data=tabulate_data, columns=tabulate_columns)
        print(tabulate(tabulate_df, headers='keys', tablefmt='grid', showindex=False))
