import numpy as np
import pandas as pd
import os
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from ClassesML.Interfaces import IKinimatics
from ClassesML.Plot import Plot
from ClassesML.InfoML import PathInfoKinematics


class Euler:

    @staticmethod
    def rad2deg(rad):

        if isinstance(rad, list):
            rad = np.array(rad)

        else:
            rad = rad

        deg = rad / np.pi * 180

        return deg

    @staticmethod
    def deg2rad(deg):

        if isinstance(deg, list):
            deg = np.array(deg)

        else:
            deg = deg

        rad = deg / 180 * np.pi

        return rad

    @staticmethod
    def rotationX(angle):

        R = np.array([[1,                0,                               0],
                      [0,                math.cos(angle),  -math.sin(angle)],
                      [0,                math.sin(angle),   math.cos(angle)]])

        return R

    @staticmethod
    def rotationY(angle):

        R = np.array([[math.cos(angle),    0,                  -math.sin(angle)],
                      [0,                  1,                                 0],
                      [math.sin(angle),    0,                   math.cos(angle)]])

        return R

    @staticmethod
    def rotationZ(angle):

        R = np.array([[math.cos(angle),   -math.sin(angle),                 0],
                      [math.sin(angle),   math.cos(angle),                  0],
                      [0,                 0,                                1]])

        return R

    @staticmethod
    def euler_angles_to_rotation_matrix(euler, seq="XYZ", to_rad=True):

        if to_rad:

            euler = Euler.deg2rad(euler)

        if seq == "XYZ":

            R = Euler.rotationX(euler[0]) @ Euler.rotationY(euler[1]) @ Euler.rotationZ(euler[2])

        elif seq == "XZY":

            R = Euler.rotationX(euler[0]) @ Euler.rotationZ(euler[1]) @ Euler.rotationY(euler[2])

        elif seq == "YXZ":

            R = Euler.rotationY(euler[0]) @ Euler.rotationX(euler[1]) @ Euler.rotationZ(euler[2])

        elif seq == "YZX":

            R = Euler.rotationY(euler[0]) @ Euler.rotationZ(euler[1]) @ Euler.rotationX(euler[2])

        elif seq == "ZXY":

            R = Euler.rotationZ(euler[0]) @ Euler.rotationX(euler[1]) @ Euler.rotationY(euler[2])

        elif seq == "ZYX":

            R = Euler.rotationZ(euler[0]) @ Euler.rotationY(euler[1]) @ Euler.rotationX(euler[2])

        return R


class Rotation:

    @staticmethod
    def to_euler_angles(rotation_matrix, seq="XYZ", to_degrees: bool = True):

        if seq == "XYZ":

            beta = np.arctan2(rotation_matrix[0, 2],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))

            alpha = np.arctan2(-rotation_matrix[1, 2] / np.cos(beta),
                               rotation_matrix[2, 2] / np.cos(beta))

            gamma = np.arctan2(-rotation_matrix[0, 1] / np.cos(beta),
                               rotation_matrix[0, 0] / np.cos(beta))

        elif seq == "XZY":

            beta = np.arctan2(-rotation_matrix[0, 1],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 2] ** 2))

            alpha = np.arctan2(rotation_matrix[2, 1] / np.cos(beta),
                               rotation_matrix[1, 1] / np.cos(beta))

            gamma = np.arctan2(rotation_matrix[0, 2] / np.cos(beta),
                               rotation_matrix[0, 0] / np.cos(beta))

        elif seq == "ZXY":

            beta = np.arctan2(rotation_matrix[2, 1],
                              np.sqrt(rotation_matrix[2, 0] ** 2 + rotation_matrix[2, 2] ** 2))

            alpha = np.arctan2(-rotation_matrix[0, 1] / np.cos(beta),
                               rotation_matrix[1, 1] / np.cos(beta))

            gamma = np.arctan2(-rotation_matrix[2, 0] / np.cos(beta),
                               rotation_matrix[2, 2] / np.cos(beta))

        elif seq == "ZYX":

            beta = np.arctan2(-rotation_matrix[2, 0],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))

            alpha = np.arctan2(rotation_matrix[1, 0] / np.cos(beta),
                               rotation_matrix[0, 0] / np.cos(beta))

            gamma = np.arctan2(rotation_matrix[2, 1] / np.cos(beta),
                               rotation_matrix[2, 2] / np.cos(beta))

        euler = np.array([alpha, beta, gamma])

        if to_degrees:
            euler = Euler.rad2deg(euler)

        return euler

    @staticmethod
    def compute_relative_rotation_matrix(parent, child):

        rotation_matrix = np.dot(child, np.linalg.inv(parent))

        return rotation_matrix

def one_hots(labels=''):
    new_word_id = 0
    words = []
    dictionary = {}
    labels = labels[:, np.newaxis]

    # sort by alphabetical order
    labels_sorted = np.sort(labels, axis=0)

    for word in labels_sorted:
        if word[0] not in dictionary:
            dictionary[word[0]] = new_word_id
            new_word_id += 1

    for word in labels:
        words.append(dictionary[word[0]])

    one_hots = to_categorical(words)

    return dictionary, one_hots

class HandFeatures():

    @staticmethod
    def compute_distance_EF(thumb_EF, index_EF, middle_EF, ring_EF, pinky_EF, palm):

        # EF relative distance
        distance1 = np.linalg.norm((thumb_EF - index_EF), axis=1)[:, np.newaxis]
        distance2 = np.linalg.norm((index_EF - middle_EF), axis=1)[:, np.newaxis]
        distance3 = np.linalg.norm((middle_EF - ring_EF), axis=1)[:, np.newaxis]
        distance4 = np.linalg.norm((ring_EF - pinky_EF), axis=1)[:, np.newaxis]

        # EF to palm relative distance
        distance5 = np.linalg.norm((thumb_EF - palm), axis=1)[:, np.newaxis]
        distance6 = np.linalg.norm((index_EF - palm), axis=1)[:, np.newaxis]
        distance7 = np.linalg.norm((middle_EF - palm), axis=1)[:, np.newaxis]
        distance8 = np.linalg.norm((ring_EF - palm), axis=1)[:, np.newaxis]
        distance9 = np.linalg.norm((pinky_EF - palm), axis=1)[:, np.newaxis]

        distances = np.concatenate((distance1, distance2, distance3,
                                    distance4, distance5, distance6,
                                    distance7, distance8, distance9), axis=1)

        return distances

    @staticmethod
    def compute_forearm_coordinate(hand_radius, wrist, elbow, display = False):

        # Y: Line connecting ulna hand to the elbow (pointing proximally)
        Y = (elbow - wrist)
        Y_norm = np.linalg.norm(Y, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Y = Y / Y_norm

        # X: Line perpendicular to the plane formed by ulna hand, radial hand and elbow (pointing forward)
        vec1 = (hand_radius - wrist)
        vec1_norm = np.linalg.norm(vec1, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec1 = vec1 / vec1_norm

        X = np.cross(Y, vec1)
        X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis].repeat(3, axis=1)
        X = X / X_norm

        # Z: The line perpendicular to X and Y (pointing to the right)
        Z = np.cross(X, Y)
        Z_norm = np.linalg.norm(Z, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Z = Z / Z_norm

        forearm_coordinate = np.concatenate([X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]], axis=2)

        if display:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            X = elbow[0, 0]
            Y = elbow[0, 1]
            Z = elbow[0, 2]

            ax.quiver(X, Y, Z, forearm_coordinate[0, 0, 0], forearm_coordinate[0, 0, 1], forearm_coordinate[0, 0, 2], color='r')
            ax.quiver(X, Y, Z, forearm_coordinate[1, 1, 0], forearm_coordinate[1, 1, 1], forearm_coordinate[1, 1, 2], color='b')
            ax.quiver(X, Y, Z, forearm_coordinate[2, 2, 0], forearm_coordinate[2, 2, 1], forearm_coordinate[2, 2, 2], color='gr')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        return forearm_coordinate

    @staticmethod
    def compute_hand_coordinate(index_proximal, pinky_proximal, palm, display = False):

        # Z: Line connecting the index and pinky base (pointing right)
        Z = (index_proximal - pinky_proximal)
        print(Z, 'Z')
        Z_norm = np.linalg.norm(Z, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Z = Z / Z_norm
        print(Z, 'ZZ')

        # X: Line perpendicular to the plane formed by index, pinky base and palm (pointing forward)
        vec1 = (index_proximal - palm)
        vec1_norm = np.linalg.norm(vec1, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec1 = vec1 / vec1_norm
        vec2 = (pinky_proximal - palm)
        vec2_norm = np.linalg.norm(vec2, axis=1)[:, np.newaxis].repeat(3, axis=1)
        vec2 = vec2 / vec2_norm

        X = np.cross(vec1, vec2)
        print(X, 'X')
        X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis].repeat(3, axis=1)
        X = X / X_norm
        print(X, 'XX')

        # Y: The line perpendicular to Z and X (pointing proximally)
        Y = np.cross(Z, X)
        print(Y, 'Y')
        Y_norm = np.linalg.norm(Y, axis=1)[:, np.newaxis].repeat(3, axis=1)
        Y = Y / Y_norm
        print(Y, 'YY')

        hand_coordinate = np.concatenate([X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]], axis=2)

        if display:

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            X = pinky_proximal[0, 0]
            Y = pinky_proximal[0, 1]
            Z = pinky_proximal[0, 2]

            ax.quiver(X, Y, Z, hand_coordinate[0, 0, 0], hand_coordinate[0, 0, 1], hand_coordinate[0, 0, 2], color='r')
            ax.quiver(X, Y, Z, hand_coordinate[1, 1, 0], hand_coordinate[1, 1, 1], hand_coordinate[1, 1, 2], color='b')
            ax.quiver(X, Y, Z, hand_coordinate[2, 2, 0], hand_coordinate[2, 2, 1], hand_coordinate[2, 2, 2], color='gr')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        return hand_coordinate

    @staticmethod
    def compute_finger_joint_local_position(df_input, hand_c_L, hand_c_R):

        input = df_input.as_matrix()
        input_L = input[:, :input.shape[1]//2]
        input_R = input[:, input.shape[1]//2:]
        input_local_L = input_L.copy()
        input_local_R = input_R.copy()

        for t in range(input_L.shape[0]):

            temp_L = np.array([hand_c_L[t, :, 0], hand_c_L[t, :, 1], hand_c_L[t, :, 2]]).transpose()
            temp_R = np.array([hand_c_R[t, :, 0], hand_c_R[t, :, 1], hand_c_R[t, :, 2]]).transpose()

            for i in range(input_L.shape[1]//3):

                input_local_L[t, i*3:i*3+3] = np.dot(temp_L.transpose(), input_L[t, i*3:i*3+3])
                input_local_R[t, i*3:i*3+3] = np.dot(temp_R.transpose(), input_R[t, i*3:i*3+3])

        input_local = np.concatenate([input_local_L, input_local_R], axis=1)

        df_input_local = pd.DataFrame(input_local, columns=df_input.columns.values)

        thumb_local_LR = df_input_local.filter(like="thumb").as_matrix()
        index_local_LR = df_input_local.filter(like="index").as_matrix()
        middle_local_LR = df_input_local.filter(like="middle").as_matrix()
        ring_local_LR = df_input_local.filter(like="ring").as_matrix()
        pinky_local_LR = df_input_local.filter(like="pinky").as_matrix()

        locals = np.concatenate([thumb_local_LR,
                                 index_local_LR,
                                 middle_local_LR,
                                 ring_local_LR,
                                 pinky_local_LR])

        return locals

    @staticmethod
    def compute_finger_flexion_angle(EF, D, M, P, display = False):

        EF_D = EF - D
        D_M = D - M
        M_P = M - P

        EF_D_length = np.linalg.norm((EF_D), axis=1)[:, np.newaxis]
        D_M_length = np.linalg.norm((D_M), axis=1)[:, np.newaxis]
        M_P_length = np.linalg.norm((M_P), axis=1)[:, np.newaxis]

        angle1 = np.empty([EF.shape[0], 1])
        angle2 = np.empty([EF.shape[0], 1])

        for i in range(EF.shape[0]):
            angle1[i, 0] = np.arccos(
                np.dot(EF_D[i, :], D_M[i, :]) / (np.linalg.norm(EF_D[i, :]) * np.linalg.norm(D_M[i, :])))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            angle2[i, 0] = np.arccos(
                np.dot(D_M[i, :], M_P[i, :]) / (np.linalg.norm(D_M[i, :]) * np.linalg.norm(M_P[i, :])))
            angle2[np.isnan(angle2)] = 0
            angle2[i, 0] = math.degrees(angle2[i, 0])

        if display:
            plt.plot(angle1, label='finger_angle_D_M')
            plt.plot(angle2, label='finger_angle_M_P')
            plt.legend()

        return angle1, angle2

    @staticmethod
    def compute_thumb_angle_y(thumb_EF, thumb_D, thumb_P):

        EF_P = thumb_EF - thumb_D
        D_P = thumb_D - thumb_P

        angle1, angle2 = np.empty([EF_P.shape[0], 1]), np.empty([EF_P.shape[0], 1])

        for i in range(thumb_EF.shape[0]):

            angle1[i, 0] = np.arccos(np.dot(EF_P[i, :], np.array([0, 1, 0])) / (np.linalg.norm(EF_P[i, :]) * np.linalg.norm(np.array([0, 1, 0]))))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            angle2[i, 0] = np.arccos(np.dot(D_P[i, :], np.array([0, 1, 0])) / (np.linalg.norm(D_P[i, :]) * np.linalg.norm(np.array([0, 1, 0]))))
            angle2[np.isnan(angle2)] = 0
            angle2[i, 0] = math.degrees(angle2[i, 0])

        return angle1, angle2

    @staticmethod
    def compute_1st_flexion_angle(M, P, hand_c, display = False):

        M_P = P - M

        angle1 = np.empty([M_P.shape[0], 1])

        for i in range(M_P.shape[0]):
            # Express the vector (D_M) in the hand reference system
            R_hand_sub = np.array([hand_c[i, :, 0], hand_c[i, :, 1], hand_c[i, :, 2]])
            D_M_local = np.dot(R_hand_sub.transpose(), M_P[i, :])

            D_M_local_YZ = D_M_local.copy()
            D_M_local_YZ[0] = 0
            D_M_local_YZ = D_M_local_YZ / np.linalg.norm(D_M_local_YZ)

            R_hand_sub_Y = R_hand_sub[:, 1]

            angle1[i, 0] = np.arccos(
                np.dot(D_M_local_YZ, R_hand_sub_Y) / (np.linalg.norm(D_M_local_YZ) * np.linalg.norm(R_hand_sub_Y)))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            '''
            ax.quiver(X, Y, Z, R_hand_sub[0, 0], R_hand_sub[0, 1], R_hand_sub[0, 2], color = 'r')
            ax.quiver(X, Y, Z, R_hand_sub[1, 0], R_hand_sub[1, 1], R_hand_sub[1, 2], color = 'b')
            ax.quiver(X, Y, Z, R_hand_sub[2, 0], R_hand_sub[2, 1], R_hand_sub[2, 2], color = 'gr')

            ax.quiver(X, Y, Z, D_M_local_YZ[0], D_M_local_YZ[1], D_M_local_YZ[2], color = 'k')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            '''

        if display:
            plt.plot(angle1, label='finger_angle_abduction')
            plt.legend()

        return angle1

    @staticmethod
    def compute_finger_abduction_angle(M, P, hand_c, display = False):

        M_P = P - M

        angle1 = np.empty([M_P.shape[0], 1])

        for i in range(M_P.shape[0]):
            # Express the vector (D_M) in the hand reference system
            R_hand_sub = np.array([hand_c[i, :, 0], hand_c[i, :, 1], hand_c[i, :, 2]])
            D_M_local = np.dot(R_hand_sub.transpose(), M_P[i, :])

            D_M_local_XY = D_M_local.copy()
            D_M_local_XY[2] = 0
            D_M_local_XY = D_M_local_XY / np.linalg.norm(D_M_local_XY)

            R_hand_sub_Y = R_hand_sub[:, 1]

            angle1[i, 0] = np.arccos(
                np.dot(D_M_local_XY, R_hand_sub_Y) / (np.linalg.norm(D_M_local_XY) * np.linalg.norm(R_hand_sub_Y)))
            angle1[np.isnan(angle1)] = 0
            angle1[i, 0] = math.degrees(angle1[i, 0])

            '''
            ax.quiver(X, Y, Z, R_hand_sub[0, 0], R_hand_sub[0, 1], R_hand_sub[0, 2], color = 'r')
            ax.quiver(X, Y, Z, R_hand_sub[1, 0], R_hand_sub[1, 1], R_hand_sub[1, 2], color = 'b')
            ax.quiver(X, Y, Z, R_hand_sub[2, 0], R_hand_sub[2, 1], R_hand_sub[2, 2], color = 'gr')

            ax.quiver(X, Y, Z, D_M_local_XY[0], D_M_local_XY[1], D_M_local_XY[2], color = 'k')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            '''

        if display:
            plt.plot(angle1, label='finger_angle_abduction')
            plt.legend()

        return angle1

    @staticmethod
    def compute_thumb_flexion_angle(EF, D, P, display = False):

        EF_D = EF - D
        D_P = D - P

        EF_D_length = np.linalg.norm((EF_D), axis=1)[:, np.newaxis]
        D_P_length = np.linalg.norm((D_P), axis=1)[:, np.newaxis]

        angle = np.empty([EF.shape[0], 1])

        for i in range(EF.shape[0]):
            angle[i, 0] = np.arccos(
                np.dot(EF_D[i, :], D_P[i, :]) / (np.linalg.norm(EF_D[i, :]) * np.linalg.norm(D_P[i, :])))
            angle[np.isnan(angle)] = 0
            angle[i, 0] = math.degrees(angle[i, 0])

        if display:
            plt.plot(angle, label='thumb_flexion_angle_D_P')

        return angle

    @staticmethod
    def compute_rotation_matrix(c, seq="ZYX"):

        angle1, angle2, angle3 = np.empty([c.shape[0], 1]), np.empty([c.shape[0], 1]), np.empty([c.shape[0], 1])

        for i in range(0, c.shape[0]):

            c_sub = np.array([c[i, :, 0], c[i, :, 1], c[i, :, 2]])
            
            euler = Rotation.to_euler_angles(rotation_matrix=c_sub, seq=seq, to_degrees=True)

            angle1[i, :] = euler[0]
            angle2[i, :] = euler[1]
            angle3[i, :] = euler[2]

        return angle1, angle2, angle3

    @staticmethod
    def compute_rerative_rotation_matrix(c1, c2, seq="ZYX", display=False, inverted=False):

        angle1, angle2, angle3 = np.empty([c1.shape[0], 1]), np.empty([c1.shape[0], 1]), np.empty([c1.shape[0], 1])

        for i in range(0, c1.shape[0]):
            c1_sub = np.array([c1[i, :, 0], c1[i, :, 1], c1[i, :, 2]]).transpose()
            c2_sub = np.array([c2[i, :, 0], c2[i, :, 1], c2[i, :, 2]])

            if inverted:
                c1_sub = np.array([c2[i, :, 0], c2[i, :, 1], c2[i, :, 2]]).transpose()
                c2_sub = np.array([c1[i, :, 0], c1[i, :, 1], c1[i, :, 2]])

            r = np.matmul(c1_sub, c2_sub)
            if r[1, 2] > 1:
                r[1, 2] = 1

            euler = Rotation.to_euler_angles(rotation_matrix=r, seq=seq, to_degrees=True)

            angle1[i, :] = euler[0]
            angle2[i, :] = euler[1]
            angle3[i, :] = euler[2]

        if display:
            plt.plot(angle1, label='X')
            plt.plot(angle2, label='Y')
            plt.plot(angle3, label='Z')
            plt.legend()

        return angle1, angle2, angle3

    @staticmethod
    def plot_hands(df_input: pd.DataFrame):

        joint_names = [x[:-2] for x in df_input.columns.values]
        input = df_input.as_matrix()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(2, input.shape[1]//3):

            x = input[0, i*3]
            y = input[0, i*3 + 1]
            z = input[0, i*3 + 2]

            ax.scatter(x, y, z)
            ax.text(x, y, z, joint_names[i*3])

    @staticmethod
    def display_hand_info(df_input: pd.DataFrame):

        pass

class Kinematics_both_hands(IKinimatics):

    coodinate_forearmR: np.ndarray
    coodinate_forearmL: np.ndarray
    coodinate_handR: np.ndarray
    coodinate_handL: np.ndarray

    _df: pd.DataFrame

    def __init__(self, df_input: pd.DataFrame, display: bool = False, drop_corr_features: bool = False):

        self.label = df_input.values[:, 1]
        self._target_names = np.unique(self.label)
        self._label_encoding()

        self.df_input = df_input

        self._filtering_data()
        self._generate_coordinate()
        self.kinematics_dict = {}
        self._generate_kinematics(display=display)
        plot = Plot()
        pathInfoKinematics = PathInfoKinematics()

        if (drop_corr_features):

            # Create correlation matrix
            corr_matrix = self._df.corr().abs()
            self._fig_corr = plot.plot_heatmap_corr_matrix(corr_matrix=corr_matrix)
            plot.save_figure(fig=self._fig_corr, path=pathInfoKinematics.path_corr_matrix,
                             figure_name="correlation_kinematics_both_hands")

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # Find index of feature columns with correlation greater than 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

            # Drop features
            self._df = self._df.drop(self._df[to_drop], axis=1)

        self._X = self._df.values
        self._feature_names = self._df.columns.tolist()

    @property
    def df(self):
        return self._df

    @property
    def target_names(self):
        return self._target_names

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def y(self):
        return self._y

    @property
    def X(self):
        return self._X

    def _filtering_data(self):

        self.thumb_EF_R = self.df_input.filter(like="thumb_EF_R").values
        self.thumb_D_R = self.df_input.filter(like="thumb_D_R").values
        self.thumb_P_R = self.df_input.filter(like="thumb_P_R").values

        self.index_EF_R = self.df_input.filter(like="index_EF_R").values
        self.index_D_R = self.df_input.filter(like="index_D_R").values
        self.index_M_R = self.df_input.filter(like="index_M_R").values
        self.index_P_R = self.df_input.filter(like="index_P_R").values

        self.middle_EF_R = self.df_input.filter(like="middle_EF_R").values
        self.middle_D_R = self.df_input.filter(like="middle_D_R").values
        self.middle_M_R = self.df_input.filter(like="middle_M_R").values
        self.middle_P_R = self.df_input.filter(like="middle_P_R").values

        self.ring_EF_R = self.df_input.filter(like="ring_EF_R").values
        self.ring_D_R = self.df_input.filter(like="ring_D_R").values
        self.ring_M_R = self.df_input.filter(like="ring_M_R").values
        self.ring_P_R = self.df_input.filter(like="ring_P_R").values

        self.pinky_EF_R = self.df_input.filter(like="pinky_EF_R").values
        self.pinky_D_R = self.df_input.filter(like="pinky_D_R").values
        self.pinky_M_R = self.df_input.filter(like="pinky_M_R").values
        self.pinky_P_R = self.df_input.filter(like="pinky_P_R").values

        self.thumb_EF_L = self.df_input.filter(like="thumb_EF_L").values
        self.thumb_D_L = self.df_input.filter(like="thumb_D_L").values
        self.thumb_P_L = self.df_input.filter(like="thumb_P_L").values

        self.index_EF_L = self.df_input.filter(like="index_EF_L").values
        self.index_D_L = self.df_input.filter(like="index_D_L").values
        self.index_M_L = self.df_input.filter(like="index_M_L").values
        self.index_P_L = self.df_input.filter(like="index_P_L").values

        self.middle_EF_L = self.df_input.filter(like="middle_EF_L").values
        self.middle_D_L = self.df_input.filter(like="middle_D_L").values
        self.middle_M_L = self.df_input.filter(like="middle_M_L").values
        self.middle_P_L = self.df_input.filter(like="middle_P_L").values

        self.ring_EF_L = self.df_input.filter(like="ring_EF_L").values
        self.ring_D_L = self.df_input.filter(like="ring_D_L").values
        self.ring_M_L = self.df_input.filter(like="ring_M_L").values
        self.ring_P_L = self.df_input.filter(like="ring_P_L").values

        self.pinky_EF_L = self.df_input.filter(like="pinky_EF_L").values
        self.pinky_D_L = self.df_input.filter(like="pinky_D_L").values
        self.pinky_M_L = self.df_input.filter(like="pinky_M_L").values
        self.pinky_P_L = self.df_input.filter(like="pinky_P_L").values

        self.palm_R = self.df_input.filter(like="palm_R").values
        self.palm_L = self.df_input.filter(like="palm_L").values

        self.hand_radius_R = self.df_input.filter(like="hand_radius_R").values
        self.wrist_R = self.df_input.filter(like="wrist_R").values
        self.elbow_R = self.df_input.filter(like="elbow_R").values

        self.hand_radius_L = self.df_input.filter(like="hand_radius_L").values
        self.wrist_L = self.df_input.filter(like="wrist_L").values
        self.elbow_L = self.df_input.filter(like="elbow_L").values

    def _label_encoding(self):
        le = LabelEncoder()
        self._y = le.fit_transform(self.label)

    def _generate_distance(self):

        # distance between the end-effector and the end-effectors with the palm
        distanceR = HandFeatures.compute_distance_EF(thumb_EF=self.thumb_EF_R, index_EF=self.index_EF_R,
                                                     middle_EF=self.middle_EF_R, ring_EF=self.ring_EF_R,
                                                     pinky_EF=self.pinky_EF_R, palm=self.palm_R)

        distanceL = HandFeatures.compute_distance_EF(thumb_EF=self.thumb_EF_L, index_EF=self.index_EF_L,
                                                     middle_EF=self.middle_EF_L, ring_EF=self.ring_EF_L,
                                                     pinky_EF=self.pinky_EF_L, palm=self.palm_L)

        return distanceR, distanceL

    def _generate_coordinate(self):

        # Right Forearm reference system
        self.coodinate_forearmR = HandFeatures.compute_forearm_coordinate(hand_radius=self.hand_radius_R,
                                                                          wrist=self.wrist_R,
                                                                          elbow=self.elbow_R, display=False)

        # Left Forearm reference system
        self.coodinate_forearmL = HandFeatures.compute_forearm_coordinate(hand_radius=self.hand_radius_L,
                                                                          wrist=self.wrist_L,
                                                                          elbow=self.elbow_L, display=False)

        # Right Hand reference system
        self.coodinate_handR = HandFeatures.compute_hand_coordinate(index_proximal=self.index_M_R,
                                                                    pinky_proximal=self.pinky_M_R,
                                                                    palm=self.palm_R, display=False)

        # Left Hand reference system
        self.coodinate_handL = HandFeatures.compute_hand_coordinate(index_proximal=self.index_M_L,
                                                                    pinky_proximal=self.pinky_M_L,
                                                                    palm=self.palm_L, display=False)

    def _generate_rotation_matrix(self):

        # Rotation matrix and angle between the two forearms ->
        angle1, angle2, angle3 = HandFeatures.compute_rerative_rotation_matrix(c1=self.coodinate_forearmR,
                                                                               c2=self.coodinate_forearmL,
                                                                               display=False)

        # omega1 = np.concatenate((angle1, angle2, angle3), axis=1)

        # Rotation matrix and angle between the elbow and the forearm right
        angle4, angle5, angle6 = HandFeatures.compute_rerative_rotation_matrix(c1=self.coodinate_forearmR,
                                                                               c2=self.coodinate_handR,
                                                                               display=False)

        # omega2 = np.concatenate((angle4, angle5, angle6), axis=1)

        # Rotation matrix and angle between the elbow and the forearm left
        angle7, angle8, angle9 = HandFeatures.compute_rerative_rotation_matrix(c1=self.coodinate_forearmL,
                                                                               c2=self.coodinate_handL,
                                                                               display=False)

        # omega3 = np.concatenate((angle7, angle8, angle9), axis=1)

        omega = np.concatenate((angle1, angle2, angle3,
                                angle4, angle5, angle6,
                                angle7, angle8, angle9), axis=1)

        return omega

    def _generate_flexion_angle(self):

        thumb_flexionR = HandFeatures.compute_thumb_flexion_angle(EF=self.thumb_EF_R,
                                                                  D=self.thumb_D_R,
                                                                  P=self.thumb_P_R, display=False)

        index_flexionR1, index_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.index_EF_R,
                                                                                     D=self.index_D_R,
                                                                                     M=self.index_M_R,
                                                                                     P=self.index_P_R, display=False)

        middle_flexionR1, middle_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.middle_EF_R,
                                                                                       D=self.middle_D_R,
                                                                                       M=self.middle_M_R,
                                                                                       P=self.middle_P_R, display=False)

        ring_flexionR1, ring_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.ring_EF_R,
                                                                                   D=self.ring_D_R,
                                                                                   M=self.ring_M_R,
                                                                                   P=self.ring_P_R, display=False)

        pinky_flexionR1, pinky_flexionR2 = HandFeatures.compute_finger_flexion_angle(EF=self.pinky_EF_R,
                                                                                     D=self.pinky_D_R,
                                                                                     M=self.pinky_M_R,
                                                                                     P=self.pinky_P_R, display=False)

        theta_R = np.concatenate((thumb_flexionR,
                                  index_flexionR1,
                                  middle_flexionR1,
                                  ring_flexionR1,
                                  pinky_flexionR1), axis=1)

        beta_R = np.concatenate((index_flexionR2,
                                 middle_flexionR2,
                                 ring_flexionR2,
                                 pinky_flexionR2), axis=1)


        thumb_flexionL = HandFeatures.compute_thumb_flexion_angle(EF=self.thumb_EF_L,
                                                                  D=self.thumb_D_L,
                                                                  P=self.thumb_P_L, display=False)

        index_flexionL1, index_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.index_EF_L,
                                                                                     D=self.index_D_L,
                                                                                     M=self.index_M_L,
                                                                                     P=self.index_P_L, display=False)

        middle_flexionL1, middle_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.middle_EF_L,
                                                                                       D=self.middle_D_L,
                                                                                       M=self.middle_M_L,
                                                                                       P=self.middle_P_L, display=False)

        ring_flexionL1, ring_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.ring_EF_L,
                                                                                   D=self.ring_D_L,
                                                                                   M=self.ring_M_L,
                                                                                   P=self.ring_P_L, display=False)

        pinky_flexionL1, pinky_flexionL2 = HandFeatures.compute_finger_flexion_angle(EF=self.pinky_EF_L,
                                                                                     D=self.pinky_D_L,
                                                                                     M=self.pinky_M_L,
                                                                                     P=self.pinky_P_L, display=False)

        theta_L = np.concatenate((thumb_flexionL,
                                  index_flexionL1,
                                  middle_flexionL1,
                                  ring_flexionL1,
                                  pinky_flexionL1), axis=1)

        beta_L = np.concatenate((index_flexionL2,
                                 middle_flexionL2,
                                 ring_flexionL2,
                                 pinky_flexionL2), axis=1)

        return theta_R, beta_R, theta_L, beta_L

    def _generate_1st_flexion_angle(self):

        index_1st_flexionR = HandFeatures.compute_1st_flexion_angle(M=self.index_M_R, P=self.index_P_R,
                                                                       hand_c=self.coodinate_handR, display=True)
        middle_1st_flexionR = HandFeatures.compute_1st_flexion_angle(M=self.middle_M_R, P=self.middle_P_R,
                                                                        hand_c=self.coodinate_handR, display=False)
        ring_1st_flexionR = HandFeatures.compute_1st_flexion_angle(M=self.ring_M_R, P=self.ring_P_R,
                                                                      hand_c=self.coodinate_handR, display=False)
        pinky_1st_flexionR = HandFeatures.compute_1st_flexion_angle(M=self.pinky_M_R, P=self.pinky_P_R,
                                                                       hand_c=self.coodinate_handR, display=False)

        alpha_R = np.concatenate((index_1st_flexionR,
                                  middle_1st_flexionR,
                                  ring_1st_flexionR,
                                  pinky_1st_flexionR), axis=1)

        index_1st_flexionL = HandFeatures.compute_1st_flexion_angle(M=self.index_M_L, P=self.index_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)
        middle_1st_flexionL = HandFeatures.compute_1st_flexion_angle(M=self.middle_M_L, P=self.middle_P_L,
                                                                        hand_c=self.coodinate_handL, display=False)
        ring_1st_flexionL = HandFeatures.compute_1st_flexion_angle(M=self.ring_M_L, P=self.ring_P_L,
                                                                      hand_c=self.coodinate_handL, display=False)
        pinky_1st_flexionL = HandFeatures.compute_1st_flexion_angle(M=self.pinky_M_L, P=self.pinky_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)

        alpha_L = np.concatenate((index_1st_flexionL,
                                  middle_1st_flexionL,
                                  ring_1st_flexionL,
                                  pinky_1st_flexionL), axis=1)

        return alpha_R, alpha_L

    def _generate_abduction_angle(self):

        index_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.index_M_R, P=self.index_P_R,
                                                                       hand_c=self.coodinate_handR, display=False)
        middle_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.middle_M_R, P=self.middle_P_R,
                                                                        hand_c=self.coodinate_handR, display=False)
        ring_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.ring_M_R, P=self.ring_P_R,
                                                                      hand_c=self.coodinate_handR, display=False)
        pinky_abductionR = HandFeatures.compute_finger_abduction_angle(M=self.pinky_M_R, P=self.pinky_P_R,
                                                                       hand_c=self.coodinate_handR, display=False)

        fai_R = np.concatenate((index_abductionR,
                                  middle_abductionR,
                                  ring_abductionR,
                                  pinky_abductionR), axis=1)

        index_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.index_M_L, P=self.index_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)
        middle_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.middle_M_L, P=self.middle_P_L,
                                                                        hand_c=self.coodinate_handL, display=False)
        ring_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.ring_M_L, P=self.ring_P_L,
                                                                      hand_c=self.coodinate_handL, display=False)
        pinky_abductionL = HandFeatures.compute_finger_abduction_angle(M=self.pinky_M_L, P=self.pinky_P_L,
                                                                       hand_c=self.coodinate_handL, display=False)

        fai_L = np.concatenate((index_abductionL,
                                  middle_abductionL,
                                  ring_abductionL,
                                  pinky_abductionL), axis=1)

        return fai_R, fai_L

    def _generate_kinematics(self, display: bool = False):

        distanceR, distanceL = self._generate_distance()

        for i in range(distanceR.shape[1]):
            self.kinematics_dict["d_" + str(i + 1) + "_R"] = distanceR[:, i]

        for i in range(distanceL.shape[1]):
            self.kinematics_dict["d_" + str(i + 1) + "_L"] = distanceL[:, i]

        omega = self._generate_rotation_matrix()

        for i in range(omega.shape[1]):
            self.kinematics_dict["omega_" + str(i+1)] = omega[:, i]

        # Computing of thumb and fingers angle (2 angles for the thumb and 3 angles for each finger)
        theta_R, beta_R, theta_L, beta_L = self._generate_flexion_angle()

        for i in range(theta_R.shape[1]):
            self.kinematics_dict["theta_" + str(i + 1) + "_R"] = theta_R[:, i]

        for i in range(theta_L.shape[1]):
            self.kinematics_dict["theta_" + str(i + 1) + "_L"] = theta_L[:, i]

        for i in range(beta_R.shape[1]):
            self.kinematics_dict["beta_" + str(i + 1) + "_R"] = beta_R[:, i]

        for i in range(beta_L.shape[1]):
            self.kinematics_dict["beta_" + str(i + 1) + "_L"] = beta_L[:, i]

        alpha_R, alpha_L = self._generate_1st_flexion_angle()

        for i in range(alpha_R.shape[1]):
            self.kinematics_dict["alpha_" + str(i+1) + "_R"] = alpha_R[:, i]

        for i in range(alpha_L.shape[1]):
            self.kinematics_dict["alpha_" + str(i+1) + "_L"] = alpha_L[:, i]

        fai_R, fai_L = self._generate_abduction_angle()

        for i in range(fai_R.shape[1]):
            self.kinematics_dict["fai_" + str(i+1) + "_R"] = fai_R[:, i]

        for i in range(fai_L.shape[1]):
            self.kinematics_dict["fai_" + str(i+1) + "_L"] = fai_L[:, i]

        self._df = pd.DataFrame(self.kinematics_dict)

        if display:
            print(self._df.head())

    def generate_locals(self, display):

        locals = HandFeatures.compute_finger_joint_local_position(df_input=self.df_input,
                                                                  hand_c_L=self.coodinate_handL,
                                                                  hand_c_R=self.coodinate_handR)

        if display:
            print(locals, "locals")
            print(locals.shap, "data locals shape")

        return locals

class Kinematics_one_hand(IKinimatics):

    coodinate_forearmR: np.ndarray
    coodinate_forearmL: np.ndarray
    coodinate_handR: np.ndarray
    coodinate_handL: np.ndarray

    _df: pd.DataFrame

    def __init__(self, df_input: pd.DataFrame, display: bool = False, drop_corr_features: bool = False):

        self.label = df_input.values[:, 1]
        self._target_names = np.unique(self.label)
        self._target_names = list(map(str, list(self._target_names)))
        self._label_encoding()

        self.df_input = df_input

        self._filtering_data()
        self._generate_coordinate()
        self.kinematics_dict = {}
        self._generate_kinematics(display=display)

        plot = Plot()
        pathInfoKinematics = PathInfoKinematics()

        if (drop_corr_features):

            # Create correlation matrix
            corr_matrix = self._df.corr().abs()
            self._fig_corr = plot.plot_heatmap_corr_matrix(corr_matrix=corr_matrix)
            plot.save_figure(fig=self._fig_corr, path=pathInfoKinematics.path_corr_matrix,
                             figure_name="correlation_kinematics_one_hand")

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # Find index of feature columns with correlation greater than 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

            # Drop features
            self._df = self._df.drop(self._df[to_drop], axis=1)

        self._X = self._df.values
        self._feature_names = self._df.columns.tolist()

    @property
    def df(self):
        return self._df

    @property
    def target_names(self):
        return self._target_names

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def y(self):
        return self._y

    @property
    def X(self):
        return self._X

    def _filtering_data(self):

        self.thumb_EF = self.df_input.filter(like="thumb_EF").values
        self.thumb_D = self.df_input.filter(like="thumb_D").values
        self.thumb_P = self.df_input.filter(like="thumb_P").values

        self.index_EF = self.df_input.filter(like="index_EF").values
        self.index_D = self.df_input.filter(like="index_D").values
        self.index_M = self.df_input.filter(like="index_M").values
        self.index_P = self.df_input.filter(like="index_P").values

        self.middle_EF = self.df_input.filter(like="middle_EF").values
        self.middle_D = self.df_input.filter(like="middle_D").values
        self.middle_M = self.df_input.filter(like="middle_M").values
        self.middle_P = self.df_input.filter(like="middle_P").values

        self.ring_EF = self.df_input.filter(like="ring_EF").values
        self.ring_D = self.df_input.filter(like="ring_D").values
        self.ring_M = self.df_input.filter(like="ring_M").values
        self.ring_P = self.df_input.filter(like="ring_P").values

        self.pinky_EF = self.df_input.filter(like="pinky_EF").values
        self.pinky_D = self.df_input.filter(like="pinky_D").values
        self.pinky_M = self.df_input.filter(like="pinky_M").values
        self.pinky_P = self.df_input.filter(like="pinky_P").values

        self.palm = self.df_input.filter(like="palm").values

        self.hand_radius = self.df_input.filter(like="hand_radius").values
        self.wrist = self.df_input.filter(like="wrist").values
        self.elbow = self.df_input.filter(like="elbow").values

    def _label_encoding(self):
        le = LabelEncoder()
        self._y = le.fit_transform(self.label)

    def _generate_distance(self):

        # distance between the end-effector and the end-effectors with the palm
        distance = HandFeatures.compute_distance_EF(thumb_EF=self.thumb_EF, index_EF=self.index_EF,
                                                    middle_EF=self.middle_EF, ring_EF=self.ring_EF,
                                                    pinky_EF=self.pinky_EF, palm=self.palm)

        return distance

    def _generate_thumb_angle_y(self):

        angle1, angle2 = HandFeatures.compute_thumb_angle_y(thumb_EF=self.thumb_EF,
                                                            thumb_D=self.thumb_D,
                                                            thumb_P=self.thumb_P)

        upsilon = np.concatenate((angle1, angle2), axis=1)

        return upsilon

    def _generate_coordinate(self):

        # Right Forearm reference system
        self.coordinate_forearm = HandFeatures.compute_forearm_coordinate(hand_radius=self.hand_radius,
                                                                          wrist=self.wrist,
                                                                          elbow=self.elbow, display=False)

        # Right Hand reference system
        self.coordinate_hand = HandFeatures.compute_hand_coordinate(index_proximal=self.index_M,
                                                                    pinky_proximal=self.pinky_M,
                                                                    palm=self.palm, display=False)

    def _generate_rotation_matrix(self):

        # Rotation matrix and angle between the elbow and the forearm
        angle1, angle2, angle3 = HandFeatures.compute_rerative_rotation_matrix(c1=self.coordinate_forearm,
                                                                               c2=self.coordinate_hand,
                                                                               seq="ZYX",
                                                                               display=False)

        angle4, angle5, angle6 = HandFeatures.compute_rotation_matrix(c=self.coordinate_forearm,
                                                                      seq="ZYX")

        angle7, angle8, angle9 = HandFeatures.compute_rotation_matrix(c=self.coordinate_hand,
                                                                      seq="ZYX")

        omega = np.concatenate((angle1, angle2, angle3,
                                angle4, angle5, angle6,
                                angle7, angle8, angle9), axis=1)

        return omega

    def _generate_flexion_angle(self):

        thumb_flexion = HandFeatures.compute_thumb_flexion_angle(EF=self.thumb_EF,
                                                                 D=self.thumb_D,
                                                                 P=self.thumb_P, display=False)

        index_flexion1, index_flexion2 = HandFeatures.compute_finger_flexion_angle(EF=self.index_EF,
                                                                                   D=self.index_D,
                                                                                   M=self.index_M,
                                                                                   P=self.index_P, display=False)

        middle_flexion1, middle_flexion2 = HandFeatures.compute_finger_flexion_angle(EF=self.middle_EF,
                                                                                     D=self.middle_D,
                                                                                     M=self.middle_M,
                                                                                     P=self.middle_P, display=False)

        ring_flexion1, ring_flexion2 = HandFeatures.compute_finger_flexion_angle(EF=self.ring_EF,
                                                                                 D=self.ring_D,
                                                                                 M=self.ring_M,
                                                                                 P=self.ring_P, display=False)

        pinky_flexion1, pinky_flexion2 = HandFeatures.compute_finger_flexion_angle(EF=self.pinky_EF,
                                                                                   D=self.pinky_D,
                                                                                   M=self.pinky_M,
                                                                                   P=self.pinky_P, display=False)

        theta = np.concatenate((thumb_flexion,
                                index_flexion1,
                                middle_flexion1,
                                ring_flexion1,
                                pinky_flexion1), axis=1)

        beta = np.concatenate((index_flexion2,
                               middle_flexion2,
                               ring_flexion2,
                               pinky_flexion2), axis=1)

        return theta, beta

    def _generate_1st_flexion_angle(self):

        index_1st_flexion = HandFeatures.compute_1st_flexion_angle(M=self.index_M, P=self.index_P,
                                                                   hand_c=self.coordinate_hand, display=False)
        middle_1st_flexion = HandFeatures.compute_1st_flexion_angle(M=self.middle_M, P=self.middle_P,
                                                                    hand_c=self.coordinate_hand, display=False)
        ring_1st_flexion = HandFeatures.compute_1st_flexion_angle(M=self.ring_M, P=self.ring_P,
                                                                  hand_c=self.coordinate_hand, display=False)
        pinky_1st_flexion = HandFeatures.compute_1st_flexion_angle(M=self.pinky_M, P=self.pinky_P,
                                                                   hand_c=self.coordinate_hand, display=False)

        alpha = np.concatenate((index_1st_flexion,
                                middle_1st_flexion,
                                ring_1st_flexion,
                                pinky_1st_flexion), axis=1)

        return alpha

    def _generate_abduction_angle(self):

        index_abduction = HandFeatures.compute_finger_abduction_angle(M=self.index_M, P=self.index_P,
                                                                      hand_c=self.coordinate_hand, display=False)
        middle_abduction = HandFeatures.compute_finger_abduction_angle(M=self.middle_M, P=self.middle_P,
                                                                       hand_c=self.coordinate_hand, display=False)
        ring_abduction = HandFeatures.compute_finger_abduction_angle(M=self.ring_M, P=self.ring_P,
                                                                     hand_c=self.coordinate_hand, display=False)
        pinky_abduction = HandFeatures.compute_finger_abduction_angle(M=self.pinky_M, P=self.pinky_P,
                                                                      hand_c=self.coordinate_hand, display=False)

        fai = np.concatenate((index_abduction,
                              middle_abduction,
                              ring_abduction,
                              pinky_abduction), axis=1)

        return fai

    def _generate_kinematics(self, display: bool = False):

        distance = self._generate_distance()

        for i in range(distance.shape[1]):
            self.kinematics_dict["d_" + str(i + 1)] = distance[:, i]

        omega = self._generate_rotation_matrix()

        for i in range(omega.shape[1]):
            self.kinematics_dict["omega_" + str(i+1)] = omega[:, i]

        # Computing of thumb and fingers angle (2 angles for the thumb and 3 angles for each finger)
        theta, beta = self._generate_flexion_angle()

        for i in range(theta.shape[1]):
            self.kinematics_dict["theta_" + str(i + 1)] = theta[:, i]

        for i in range(beta.shape[1]):
            self.kinematics_dict["beta_" + str(i + 1)] = beta[:, i]

        alpha = self._generate_1st_flexion_angle()

        for i in range(alpha.shape[1]):
            self.kinematics_dict["alpha_" + str(i+1)] = alpha[:, i]

        fai = self._generate_abduction_angle()

        for i in range(fai.shape[1]):
            self.kinematics_dict["fai_" + str(i+1)] = fai[:, i]

        upsilon = self._generate_thumb_angle_y()

        for i in range(upsilon.shape[1]):
            self.kinematics_dict["upsilon_" + str(i+1)] = upsilon[:, i]

        self._df = pd.DataFrame(self.kinematics_dict)

        if display:
            print(self._df.head())
