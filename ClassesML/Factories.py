from ClassesML.Interfaces import *
from ClassesML.Models import *
from ClassesML.DataSets import *
from ClassesML.InfoML import PathInfoML
import pandas as pd

class ModelFactory():

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model
        self.model_name = hyper_model["model_name"]

    def create(self) -> IModel:

        if self.model_name == Model.RF:

            model = RF(self.hyper_model)

        elif self.model_name == Model.KNN:

            model = KNN(self.hyper_model)

        elif self.model_name == Model.SVC:

            model = SV(self.hyper_model)

        elif self.model_name == Model.LR:

            model = LR(self.hyper_model)

        elif self.model_name == Model.AB:

            model = AB(self.hyper_model)

        elif self.model_name == Model.GB:

            model = GB(self.hyper_model)

        elif self.model_name == Model.CB:

            model = CB(self.hyper_model)

        elif self.model_name == Model.XGB:

            model = XGB(self.hyper_model)

        elif self.model_name == Model.LGBM:

            model = LGBM(self.hyper_model)

        elif self.model_name == Model.NB:

            model = NB(self.hyper_model)

        else:
            model = None

        ML_model = model.create()

        return ML_model

class DataSetFactory():

    def __init__(self, dataset_name: str, df_input: pd.DataFrame = None):

        self.dataset_name = dataset_name
        self.df_input = df_input

    def create(self) -> IDataSet:

        if self.dataset_name == DataSets.Iris:

            datasets = Iris(display=True)

        elif self.dataset_name == DataSets.Grasp_both:

            datasets = GraspBothHands(df_input=self.df_input, display=True)

        elif self.dataset_name == DataSets.Grasp_one:

            datasets = GraspOneHand(df_input=self.df_input, display=True)

        elif self.dataset_name == DataSets.Wine:

            datasets = Wine(display=True)

        elif self.dataset_name == DataSets.Digits:

            datasets = Digits(display=True)

        else:

            datasets = None

        return datasets
