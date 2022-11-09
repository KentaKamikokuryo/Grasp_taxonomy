import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from functools import reduce
from dtreeviz.trees import dtreeviz
from ClassesML.Models import Model
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import pydotplus
from IPython.display import Image
from graphviz import Digraph

class Plot():

    def __init__(self):

        sns.set(style='ticks', rc={"grid.linewidth": 0.1})
        sns.set_context("paper", font_scale=1)
        color = sns.color_palette("Set2", 6)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

    @staticmethod
    def show_values(pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857
        By HYRY
        '''

        pc.update_scalarmappable()
        ax = pc.axes
        # ax = pc.axes# FOR LATEST MATPLOTLIB
        # Use zip BELOW IN PYTHON 3
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    @staticmethod
    def cm2inch(*tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    @staticmethod
    def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
                correct_orientation=False, cmap='RdBu', fmt="%.2f"):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()
        # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, AUC.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        Plot.show_values(c, fmt=fmt)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            # resize
        fig = plt.gcf()
        # fig.set_size_inches(cm2inch(40, 20))
        # fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(Plot.cm2inch(figure_width, figure_height))

        return fig

    @staticmethod
    def plot_classification_report(classification_report, title="Classification report", cmap="BuPu"):
        """
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        :param classification_report:
        :param title:
        :param cmap:
        :return:
        """

        lines = classification_report.split("\n")
        classes = []
        plotMat = []
        support = []
        class_names = []

        for line in lines[2: (len(lines) - 5)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            print(v)
            plotMat.append(v)

        print('plotMat: {0}'.format(plotMat))
        print('support: {0}'.format(support))

        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = True
        fig = Plot.heatmap(np.array(plotMat),
                           title,
                           xlabel,
                           ylabel,
                           xticklabels,
                           yticklabels,
                           figure_width,
                           figure_height,
                           correct_orientation,
                           cmap=cmap)

        return fig


    @staticmethod
    def plot_bar_classification_scores(ldf: list):

        frame_scores = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), ldf).T
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        frame_scores.plot.bar(ax=ax, cmap="RdYlBu", edgecolor="black")
        ax.legend(loc = "best")
        ax.set_ylabel("Score")
        ax.set_xlabel("Models")
        ax.set_title("Cross validation model benchmark")

        return fig


    @staticmethod
    def plot_feature_imp(feature_imp: pd.Series):

        print(feature_imp)
        plt.ioff()

        fig = plt.figure(figsize=(18,18))
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle("Visualizing Importance Features")

        sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax)

        ax.set_xlabel("Feature Importance Score")
        ax.set_ylabel("Features")

        plt.subplots_adjust(left=0.12, bottom=0.06, right=0.97, top=0.945, wspace=0.2, hspace=0.2)

        return fig

    @staticmethod
    def plot_permutation_importance(ML_model, X, y, df):

        test_result = permutation_importance(estimator=ML_model,
                                             X=X,
                                             y=y,
                                             n_repeats=10)

        df_importance = pd.DataFrame(zip(df.columns, test_result["importances"].mean(axis=1)),
                                     columns=["Features", "Importance"])
        df_importance = df_importance.sort_values("Importance", ascending=False)

        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)
        sns.barplot(x="Importance", y="Features", data=df_importance, ci=None, ax=ax)
        plt.title("Permutation Importance")
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, index, model_name):

        plt.ioff()
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        matrix = confusion_matrix(y_test, y_pred)

        xlabel = 'Predicted Class'
        ylabel = 'True Class'
        title = "Confusion matrix - " + str(model_name)
        xticklabels = index
        yticklabels = index
        figure_width = 25
        figure_height = len(index) + 7
        correct_orientation = True
        fig = Plot.heatmap(np.array(matrix),
                           title,
                           xlabel,
                           ylabel,
                           xticklabels,
                           yticklabels,
                           figure_width,
                           figure_height,
                           correct_orientation,
                           cmap="BuPu",
                           fmt="%.f")

        # matrix = confusion_matrix(y_test, y_pred)
        # matrix = pd.DataFrame(matrix, columns=index, index=index)
        # sns.heatmap(matrix, annot=True, fmt="d", cmap="cool", cbar=True, ax=ax)

        return fig

    @staticmethod
    def plot_heatmap_corr_matrix(corr_matrix):

        plt.ioff()
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, annot=True, ax=ax)

        plt.title("Confusion matrix showing feature correlations")

        return fig

    @staticmethod
    def plot_tree(clf_name, clf, X, y, index, class_names, path: str, figure_name: str):

        if clf_name == Model.RF:
            tree = clf.estimators_[0]

        else:
            tree = clf

        viz = dtreeviz(tree_model=tree,
                       x_data=X,
                       y_data=y,
                       target_name="variety",
                       feature_names=index,
                       class_names=[str(i) for i in class_names],
                       orientation="LR",)

        # graph = pydotplus.graph_from_dot_data(viz)
        # graph.write_pdf(path + figure_name + ".pdf")
        # Image(graph.create_png())
        # viz.view()
        viz.save(path + figure_name + ".svg")

        # viz.save(path + figure_name + ".svg")
        # print("Figure saved to: " + path + figure_name + ".svg")
        #viz.view()

    @staticmethod
    def plot_pdp_ice(clf_name, clf, X, y, index):

        pass

    @staticmethod
    def save_figure(fig, path: str, figure_name: str, close_figure: bool = True):

        fig.savefig(path + figure_name + ".png")
        print("Figure saved to: " + path + figure_name + ".png")

        if close_figure:
            plt.close()




