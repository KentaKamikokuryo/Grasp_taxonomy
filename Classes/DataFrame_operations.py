import pandas as pd
import numpy as np
from functools import singledispatch


class DataFrame_operations():

    @staticmethod
    def compute_mean_value_list(list_dataframe: list, i_reference: int = 0, index=None, columns=None):

        if not DataFrame_operations.check_shapes_equal(list_dataframe):

            print("first argument is invalid!!")

        ndarray_values = np.stack([df.values for df in list_dataframe], axis=0)

        ndarray_mean = np.mean(ndarray_values, axis=0)

        if index is not None and columns is not None:

            df_mean = pd.DataFrame(data=ndarray_mean, index=index, columns=columns)

        else:

            df_mean = pd.DataFrame(data=ndarray_mean, index=list_dataframe[i_reference].index.values,
                                   columns=list_dataframe[i_reference].columns)

        return df_mean

    @staticmethod
    def compute_mean_value(dataframe: pd.DataFrame, n_dimension: int = 1, i_reference: int = 0, index=None, columns=None):

        list_dataframe = DataFrame_operations.to_list_dataframe(dataframe=dataframe, n_dimension=n_dimension)

        if list_dataframe is None:
            return

        df_mean = DataFrame_operations.compute_mean_value_list(list_dataframe, i_reference=i_reference, index=index,
                                                               columns=columns)

        return df_mean

    @staticmethod
    def compute_relative_value(dataframe_target, dataframe_base, base_to_origin=False):

        values_target = dataframe_target.values
        values_base = dataframe_base.values

        relative_values_target = np.empty(shape=values_target.shape)
        
        n_dimension = values_base.shape[1]

        for i in range(values_target.shape[1] // n_dimension):
            relative_values_target[:, n_dimension * i:n_dimension * i + n_dimension] \
                = values_target[:, n_dimension * i:n_dimension * i + n_dimension] - values_base

        relative_values_target = np.round(relative_values_target, decimals=3)

        df_relative_target = pd.DataFrame(data=relative_values_target, index=dataframe_target.index.values, columns=dataframe_target.columns)
        df_base = dataframe_base

        if base_to_origin:
            values_base_origin = np.zeros(shape=values_base.shape)

            df_base = pd.DataFrame(data=values_base_origin)

        return df_relative_target, df_base

    @staticmethod
    def to_list_dataframe(dataframe: pd.DataFrame, n_dimension: int = 1):

        if len(dataframe.columns) % n_dimension != 0:

            print("arguments are invalid!!")
            return None

        n_data = int(len(dataframe.columns) / n_dimension)

        list_dataframe = [dataframe.iloc[:, i * n_dimension:i * n_dimension + n_dimension] for i in range(n_data)]

        return list_dataframe

    @staticmethod
    def to_dict_dataframe(dataframe: pd.DataFrame, list_name: list, n_dimension: int = 1):

        if len(dataframe.columns) % n_dimension != 0:

            print("arguments are invalid!!")
            return None

        n_data = int(len(dataframe.columns) / n_dimension)

        if n_data != len(list_name):

            print("arguments are invalid!!")
            return None

        dict_dataframe = {list_name[i]: dataframe.iloc[:, i * n_dimension:i * n_dimension + n_dimension]
                          for i in range(n_data)}

        return dict_dataframe

    @staticmethod
    def check_shapes_equal(list_dataframe: list):

        ndarray_shapes = np.array([list(dataframe.shape) for dataframe in list_dataframe])

        unique_shapes = np.unique(ndarray_shapes, axis=0)

        if len(unique_shapes) == 1:

            return True

        else:

            return False
