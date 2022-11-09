import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

class Latent_space():

    @staticmethod
    def plot_latent_space(X_train, y_train=None, X_test=None, y_test=None, is_save=False, path_save=None):

        if is_save:

            plt.ioff()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if (X_test is not None) and (y_test is not None):

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Latent_space.create_colors(unique_class=unique_class)

            for num in unique_class:
                bool_list_train = [y_train == num]
                bool_list_train = np.array(bool_list_train)
                bool_list_train = bool_list_train.reshape(-1)

                component1_train = X_train[:, 0]
                component2_train = X_train[:, 1]
                component1_train = component1_train[bool_list_train]
                component2_train = component2_train[bool_list_train]

                ax.scatter(component1_train, component2_train, color=colors[num], label=str(num) + '_train', s=20,
                           marker='o', ec='none')

            for num in unique_class:
                bool_list_test = [y_test == num]
                bool_list_test = np.array(bool_list_test)
                bool_list_test = bool_list_test.reshape(-1)

                component1_test = X_test[:, 0]
                component2_test = X_test[:, 1]
                component1_test = component1_test[bool_list_test]
                component2_test = component2_test[bool_list_test]

                ax.scatter(component1_test, component2_test, color=colors[num], label=str(num) + '_test', s=20,
                           marker='o', ec='k', lw=0.5)

                ax.legend(fontsize=8)

        elif y_train is None:

            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, len(X_train)))

            component1_train = X_train[:, 0]
            component2_train = X_train[:, 1]

            ax.scatter(component1_train, component2_train, color=colors, s=20, marker='o', ec='k', lw=0.5)

        else:

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Latent_space.create_colors(unique_class=unique_class)

            for num in unique_class:
                bool_list = [y_train == num]
                bool_list = np.array(bool_list)
                bool_list = bool_list.reshape(-1)

                component1 = X_train[:, 0]
                component2 = X_train[:, 1]
                component1 = component1[bool_list]
                component2 = component2[bool_list]

                ax.scatter(component1, component2, color=colors[num], label=str(num), s=20, marker='o', ec='none', lw=0.5)

                ax.legend(fontsize=8)

        ax.set_xlabel('1st_component')
        ax.set_ylabel('2nd_component')

        if is_save and (path_save is not None):

            plt.savefig(path_save)
            plt.close()

        else:
            plt.show()

    @staticmethod
    def create_colors(unique_class, cmap_name="jet"):

        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, len(unique_class)))
        colors = dict(zip(unique_class, colors))

        return colors

class Utilities:

    @staticmethod
    def three_d_plot(x, y, z,path_folder_figure: str = '', save_name: str = '',save_figure: bool = True, close_figure: bool = True)\
            -> None:

        # plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
        # plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
        # plt.rcParams["font.size"] = 20  # 全体のフォントサイズが変更されます。
        # plt.rcParams['xtick.labelsize'] = 30  # 軸だけ変更されます。
        # plt.rcParams['ytick.labelsize'] = 30  # 軸だけ変更されます
        # plt.rcParams["axes.labelsize"] = 35
        # plt.rcParams['xtick.direction'] = 'in'  # x axis in
        # plt.rcParams['ytick.direction'] = 'in'  # y axis in
        # plt.rcParams['axes.linewidth'] = 1.0  # axis line width
        # plt.rcParams['axes.grid'] = True  # make grid
        # plt.rcParams["legend.fancybox"] = False  # 丸角
        # plt.rcParams["legend.framealpha"] = 0.6  # 透明度の指定、0で塗りつぶしなし
        # plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        # plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
        # plt.rcParams["legend.labelspacing"] = 0.1  # 垂直（縦）方向の距離の各凡例の距離
        # plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
        # plt.rcParams["legend.borderaxespad"] = 0.  # 凡例の端とグラフの端を合わせる
        # plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # y軸小数点以下1桁表示
        # plt.gca().xaxis.get_major_formatter().set_useOffset(False)
        # plt.locator_params(axis='y', nbins=6)  # y軸，6個以内．

        fig = plt.figure(figsize= (8, 8))

        ax = fig.add_subplot(111, projection='3d')
        data = np.stack([x, y], 0)
        # cmap = plt.cm.get_cmap('jet')
        # colors = cmap(np.linspace(0, 1, len(data)))

        ax.plot(x,y,z)

        if not os.path.exists(path_folder_figure):
            os.makedirs(path_folder_figure)

        if save_figure:
            fig.savefig(path_folder_figure + save_name)

        if close_figure:
            plt.close()