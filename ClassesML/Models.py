import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

from ClassesML.Interfaces import IModel


class Model:

    RF = "RandomForest"
    KNN = "K-NearestNeighbor"
    LR = "LogisticRegression"
    AB = "AdaBoost"
    GB = "GradientBoost"
    CB = "CatBoost"
    XGB = "XGBoost"
    LGBM = "LightGBM"
    NB = "NaiveBayes"
    SVC = "SupportVectorClassifier"

class ModelInfo:

    def __init__(self):

        self._model_names = [Model.RF,
                             Model.KNN,
                             Model.LR,
                             Model.XGB,
                             Model.LGBM,
                             Model.NB]

        # XGB, KNN, NB don't have class_weight parameter

        self._model_names = [Model.RF,
                             Model.LGBM,
                             Model.SVC,
                             Model.LR]

    @property
    def model_names(self):
        return self._model_names


class RF(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        # Number of trees in random forest
        self.n_estimators = hyper["n_estimators"]

        # Minimum number of samples required to split a node
        self.min_samples_leaf = hyper["min_samples_leaf"]

        # Criterion
        self.criterion = hyper["criterion"]

        # class weight
        self.class_weight = hyper["class_weight"]

    def create(self):

        model = RandomForestClassifier(n_estimators=self.n_estimators,
                                       criterion=self.criterion,
                                       min_samples_leaf=self.min_samples_leaf,
                                       class_weight=self.class_weight)

        return model


class SV(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        self.C = hyper["C"]

        self.kernel = hyper["kernel"]

        self.decision_function_shape = hyper["decision_function_shape"]

        self.class_weight = hyper["class_weight"]


    def create(self):

        model = SVC(C=self.C,
                    kernel=self.kernel,
                    decision_function_shape=self.decision_function_shape,
                    class_weight=self.class_weight)

        return model

class KNN(IModel):

    def __init__(self, hyper: dict):

        self.n_neighbors = hyper["n_neighbors"]

        self.n_jobs = hyper["n_jobs"]

    def create(self):

        model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                     n_jobs=self.n_jobs)

        return model



class LR(IModel):

    def __init__(self, hyper: dict):

        # Inverse of regularization strength; must be a positive float. Like in support vector machines,
        # smaller values specify stronger regularization.
        self.C = hyper["C"]

        self.random_state = hyper["random_state"]

        self.max_iter = hyper["max_iter"]

        self.class_weight = hyper["class_weight"]

    def create(self):

        model = LogisticRegression(C=self.C,
                                   random_state=self.random_state,
                                   max_iter=self.max_iter,
                                   class_weight=self.class_weight)

        return model

class AB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = AdaBoostClassifier()

        return model

class GB(IModel):

    def __init__(self, hyper: dict):

        pass

    def create(self):

        model = GradientBoostingClassifier()

        return model

class XGB(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        # Number of boosting rounds
        self.n_estimators = hyper["n_estimators"]

        # Boosting learning rate
        self.learning_rate = hyper["learning_rate"]

        # Maximum tree depth for base learners
        self.max_depth = hyper["max_depth"]

    def create(self):

        model = XGBClassifier(n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate,
                              max_depth=self.max_depth)

        return model

class LGBM(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        # Number of boosted trees to fit
        self.n_estimators = hyper["n_estimators"]

        # Boosting learning rate
        self.learning_rate = hyper["learning_rate"]

        # Maximum tree depth for base learners
        self.max_depth = hyper["max_depth"]

        # class weight
        self.class_weight = hyper["class_weight"]


    def create(self):

        model = LGBMClassifier(n_estimators=self.n_estimators,
                               learning_rate=self.learning_rate,
                               max_depth=self.max_depth,
                               class_weight=self.class_weight)

        return model

class CB(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        self.iterations = hyper["iterations"]
        self.learning_rate = hyper["learning_rate"]

    def create(self):

        model = CatBoostClassifier(iterations=self.iterations,
                                   learning_rate=self.learning_rate)

        return model

class NB(IModel):

    def __init__(self, hyper: dict):

        self.hyper = hyper

        self.var_smoothing = hyper["var_smoothing"]

    def create(self):

        model = GaussianNB(var_smoothing=self.var_smoothing)

        return model





