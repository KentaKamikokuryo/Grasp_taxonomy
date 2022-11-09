import numpy as np
import pandas as pd
import os
from ClassesML.DataSets import *
from ClassesML.Models import ModelInfo
from sklearn.metrics import classification_report
from ClassesML.Factories import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn import metrics
from ClassesML.Ranked import Ranked
from ClassesML.Hyperparameters import Hyperparameters
from ClassesML.InfoML import PathInfoML, DataInfoML
from ClassesML.Plot import Plot
from ClassesML.Models import Model
import logging as log

score_list = ["accuracy", "f1", "precision", "recall"]

def output_score(metric, y_true, y_pred):

    if metric == "accuracy":

        score = metrics.accuracy_score(y_true, y_pred)

    elif metric == "f1":

        score = metrics.f1_score(y_true, y_pred, average="weighted")

    elif metric == "recall":

        score = metrics.recall_score(y_true, y_pred, average="weighted")

    elif metric == "roc_auc":

        score = metrics.roc_auc_score(y_true, y_pred, average="weighted")

    elif metric == "precision":

        score = metrics.precision_score(y_true, y_pred, average="weighted")

    else:

        log.error("metric is not included!!!!!!")

    return score

class Manager:

    hyper_model_search: bool
    train_final_model: bool
    perform_analysis: bool
    save_model: bool
    save_results: bool

    ranked: Ranked

    def __init__(self,
                 data_name: str,
                 experiment_name: str,
                 data_info: dict,
                 metric: str,
                 list_subjects_for_test: list = None,
                 data_level: str = None):

        # Set information by dictionary
        self.test_size = data_info["test_size"]
        self.n_K_fold = data_info["n_K_fold"]
        self.k_shuffle = data_info["k_shuffle"]
        self.data_mode = data_info["data_mode"]
        self.hand_mode = data_info["hand_mode"]

        self.data_name = data_name
        self.experiment_name = experiment_name
        self.data_level = data_level
        self.list_subjects_for_test = list_subjects_for_test

        self.test_name_as_subjects = ""
        if self.list_subjects_for_test is not None:
            for test_subject in self.list_subjects_for_test:
                self.test_name_as_subjects += test_subject

        # Set metric for comparing model result
        self.metric = metric

        # Set the model name for validating
        self.modelInfo = ModelInfo()

        # Set the path for saving results
        self.pathInfoML = PathInfoML(data_name=self.data_name,
                                     experiment_name=self.experiment_name,
                                     test_subject_name=self.test_name_as_subjects,
                                     data_level=self.data_level)

        # set the data folder name
        self.dataInfoML = DataInfoML(data_mode=self.data_mode,
                                     hand_mode=self.hand_mode,
                                     data_level=self.data_level)

        # Set class for plotting results
        self.plot = Plot()

        self.__set_data()

    def __create_ML_model(self, hyper_model) -> IModel:

        fac = ModelFactory(hyper_model=hyper_model)
        ML_model = fac.create()

        return ML_model

    def __set_hyper(self, model_name):

        self.ranked = Ranked(model_name=model_name,
                             metric="f1",
                             path=self.path_search)

        if self.hyper_model_search:

            hyperParams = Hyperparameters(model_name=model_name)
            self.hyper_model_list, self.hyper_model_dict = hyperParams.generate_hypermodel(display=True)

        else:

            self.ranked.load_ranked_list(display=True)
            self.ranked.load_best_hyperparameter()
            self.hyper_model_best = self.ranked.hyperparameter_best
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted

    def __set_data(self):

        path = self.pathInfoML.path_experiment + "analysis\\" + self.dataInfoML.csv_data_name
        self.df_input = pd.read_csv(path)

        data_fac = DataSetFactory(dataset_name=self.data_name, df_input=self.df_input)
        dataset_model = data_fac.create()

        # create data-sets
        self.X, self.y, self.index, self.class_names, self.df = dataset_model.create()

        if self.list_subjects_for_test is not None:
            boolean_test = [False for _ in range(len(self.df_input))]

            for test_subject in self.list_subjects_for_test:
                temp = np.array(self.df_input["Id"].str.contains(test_subject))
                boolean_test = np.logical_or(boolean_test, temp)

            boolean_train = np.logical_not(boolean_test)

            self.X_train = self.X[boolean_train]
            self.X_test = self.X[boolean_test]
            self.y_train = self.y[boolean_train]
            self.y_test = self.y[boolean_test]

        else:

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                    self.y,
                                                                                    test_size=self.test_size,
                                                                                    shuffle=False)

        ss = preprocessing.StandardScaler()
        self.X_train = ss.fit_transform(self.X_train)
        self.X_test = ss.transform(self.X_test)

        mm = preprocessing.MinMaxScaler()
        self.X_train = mm.fit_transform(self.X_train)
        self.X_test = mm.transform(self.X_test)

    def __fit_valid(self, hyper_model):

        cv = KFold(n_splits=self.n_K_fold, shuffle=True)
        ML_model = self.__create_ML_model(hyper_model=hyper_model)

        scores = []

        for fit_index, valid_index in cv.split(X=self.X_train):

            # get train and test data
            X_fit, X_valid = self.X_train[fit_index], self.X_train[valid_index]
            y_fit, y_valid = self.y_train[fit_index], self.y_train[valid_index]

            # fit model
            ML_model.fit(X_fit, y_fit)

            # predict test data
            y_pred = ML_model.predict(X_valid)

            # loss
            metric = output_score(metric=self.metric, y_true=y_valid, y_pred=y_pred)
            scores.append(metric)

        metric_mean = np.mean(scores)
        metric_std = np.std(scores)

        return metric_mean, metric_std

    def __test(self, hyper_model, num = 0):

        ML_model = self.__create_ML_model(hyper_model=hyper_model)

        ML_model.fit(self.X_train, self.y_train)
        y_pred = ML_model.predict(self.X_test)

        score = output_score(metric=self.metric, y_true=self.y_test, y_pred=y_pred)
        scores = []
        for metric in score_list:

            score_temp = output_score(metric=metric, y_true=self.y_test, y_pred=y_pred)

            if metric == self.metric:
                score = score_temp.copy()

            scores.append(score_temp)

        tmp = pd.DataFrame({hyper_model["model_name"]: scores}, index=score_list)
        self._ldf.append(tmp)

        classification_result = classification_report(y_true=self.y_test,
                                                      y_pred=y_pred,
                                                      target_names=self.class_names)

        # For drawing classification report
        fig = self.plot.plot_classification_report(classification_report=classification_result,
                                                   title="Classification report - " + hyper_model["model_name"])
        self.plot.save_figure(fig=fig,
                              path=self.pathInfoML.path_figure_data,
                              figure_name=hyper_model["model_name"] + "_classification_report_" + self.test_name_as_subjects,
                              close_figure=True)

        # For plotting confusion matrix
        fig = self.plot.plot_confusion_matrix(y_test=self.y_test,
                                              y_pred=y_pred,
                                              index=self.class_names,
                                              model_name=hyper_model["model_name"])
        self.plot.save_figure(fig=fig,
                              path=self.pathInfoML.path_figure_data,
                              figure_name=hyper_model["model_name"] + "_confusion_matrix_" + self.test_name_as_subjects,
                              close_figure=True)

        # For plotting permutation importance
        fig = self.plot.plot_permutation_importance(ML_model=ML_model,
                                                    X=self.X_test,
                                                    y=self.y_test,
                                                    df=self.df)
        self.plot.save_figure(fig=fig,
                              path=self.pathInfoML.path_figure_data,
                              figure_name=hyper_model["model_name"] + "_permutation_imp_" + self.test_name_as_subjects,
                              close_figure=True)

        # For Plotting future importance
        if hyper_model["model_name"] == Model.LR:

            pass

        elif hyper_model["model_name"] == Model.KNN:

            pass

        elif hyper_model["model_name"] == Model.NB:

            pass

        elif hyper_model["model_name"] == Model.SVC:

            pass

        else:

            # self.plot.plot_tree(clf_name=hyper_model["model_name"],
            #                     clf=ML_model,
            #                     X=self.X_train,
            #                     y=self.y_train,
            #                     index=self.index,
            #                     path=self.pathInfoML.path_figure_data,
            #                     class_names=self.class_names,
            #                     figure_name=hyper_model["model_name"] + "_tree_viz")

            feature_imp = pd.Series(ML_model.feature_importances_, index=self.index).sort_values(ascending=False)

            fig = self.plot.plot_feature_imp(feature_imp=feature_imp)
            self.plot.save_figure(fig=fig,
                                  path=self.pathInfoML.path_figure_data,
                                  figure_name=hyper_model["model_name"] + "_feature_imp_" + self.test_name_as_subjects,
                                  close_figure=True)

        return score

    def __run_search(self):

        for model_name in self.modelInfo.model_names:

            self.path_search = self.pathInfoML.set_path_model(model_name=model_name)

            self.__set_hyper(model_name=model_name)

            for hyper_model in self.hyper_model_list:

                metric_mean, metric_std = self.__fit_valid(hyper_model=hyper_model)
                self.ranked.add(hyperparameter=hyper_model, mean=metric_mean, std=metric_std)

            self.ranked.ranked(display=True, save=self.save_best_search)
            self.ranked.save_best_hyperparameter()

            print("Run search is done on " + str(self.data_name) + "with model " + model_name)

        print("Data-sets search is done on data-sets name: " + str(self.data_name))

    def __run_comparison(self):

        ranked_comparison = Ranked(model_name="Comparison_models",
                                   metric="f1",
                                   path=self.pathInfoML.path_result_data)

        self._ldf = []

        for model_name in self.modelInfo.model_names:

            self.path_search = self.pathInfoML.set_path_model(model_name=model_name)
            self.__set_hyper(model_name=model_name)

            scores = []

            accuracy = self.__test(hyper_model=self.hyper_model_best)
            scores.append(accuracy)

            ranked_comparison.add(hyperparameter=self.hyper_model_best,
                                  mean=np.mean(scores),
                                  std=np.std(scores))

        # For plotting bar as classification scores
        fig = self.plot.plot_bar_classification_scores(ldf=self._ldf)
        self.plot.save_figure(fig=fig,
                              path=self.pathInfoML.path_figure_data,
                              figure_name="classification_scores_" + self.test_name_as_subjects,
                              close_figure=True)

        ranked_comparison.ranked(display=True, save=self.save_best_comparison)
        ranked_comparison.save_best_hyperparameter()

    def set_interface(self, interface_dict: dict):

        self.hyper_model_search = interface_dict["hyper_model_search"]
        self.save_best_search = interface_dict["save_best_search"]
        self.save_best_comparison = interface_dict["save_best_comparison"]

    def run(self):

        if self.hyper_model_search:

            self.__run_search()

        else:

            self.__run_comparison()

Is = [1]
I = 0

list_subjects_for_test = ["kami"]
data_levels = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"]
data_levels = ["L0", "L3", "L6", "L8", "L9", "L11"]



for I in Is:

    if I == 0:

        interface_dict = {"hyper_model_search": True,
                          "save_best_search": True,
                          "save_best_comparison": True}

    elif I == 1:

        interface_dict = {"hyper_model_search": False,
                          "save_best_search": True,
                          "save_best_comparison": True}

    else:

        interface_dict = {}

    dataInfo = DataInfo()
    data_names = dataInfo.get_dataset_names()
    data_info = dataInfo.get_data_info()

    for data_name in data_names:

        for data_level in data_levels:

            print("Running on" + str(data_name) + " - " + str(data_level))

            manager = Manager(data_name=data_name,
                              experiment_name="experiment_5",
                              data_info=data_info,
                              list_subjects_for_test=list_subjects_for_test,
                              metric="f1",
                              data_level=data_level)

            manager.set_interface(interface_dict=interface_dict)

            manager.run()