import math, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from Classes.Info import DataInfo1
from Classes.Data import DataFactory, Creation, DataReader
from sklearn import decomposition
from Classes.Utilities import Spline
from Classes.Factories import DataInfoFactory

# define experiment names
experiment_names = ["experiment_1", "experiment_2", "experiment_3"]
experiment_name = experiment_names[1]

# define factory and get dateInfo and take_names
fac = DataInfoFactory()
dataInfo = fac.get_Datainfo(name=experiment_name)
take_names = fac.get_take_names(name=experiment_name)

# define name of file
take_name = take_names[1]
name_file = "original"

# set contents of data information and generate data dictionary
dataInfo.set_data_info(name_take=take_name)
data_dict = dataInfo.get_data_dict(name_file=name_file)

name = "take-A-3"
name_file = "indices"
data = DataFactory(data_dict=data_dict, data_info=dataInfo)
indices = data.create()
indices_12 = indices[12]

# name_file_data = "original"
# dataInfo = DataInfo1(name_take=name, name_file=name_file_data)
# data_original = DataFactory(data_info=dataInfo)
# df_original = data_original.create()
#
# name_file_data = "original"
# dataInfo = DataInfo1(name_take=name, name_file=name_file_data)
#
# parameter = {'data_info':dataInfo,
#              'units':"Marker",
#              'parts':"PupilCenterR",
#              'position':None,
#              'rotation':None}
# dataReader = DataReader(**parameter)
# data = dataReader.getData()
#
# spline = Spline(indices=indices_12, data_info=dataInfo, df_1=data, spline=50)
