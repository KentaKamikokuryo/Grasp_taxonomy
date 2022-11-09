import math
import numpy as np
from scipy.spatial import distance
from abc import ABC, abstractmethod
from tqdm import tqdm


class Classifier(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass


class ClassifierWardOrder_N(Classifier):

    """
    The user for getting data uses two function.
    firstly, you generate the class named this.
    Then, call function named "fit" for applying TSW
    Then, if you want to get labels name, you call function named "get_labels"
    """

    def __init__(self, n_clusters: int = 2, list_save_n=None, display_mode="normal"):

        self.n_clusters_final = n_clusters

        if list_save_n is None:

            self.list_save_n = [self.n_clusters_final]

        else:

            self.list_save_n = list_save_n

        self.dict_indices_clusters = {}

        self.display_mode = display_mode

        if self.display_mode == "normal":

            self.display_func = self.__display_normal

        elif self.display_mode == "tqdm":

            self.display_func = self.__display_tqdm

        else:

            self.display_func = self.__display_normal

    def fit(self, X, y=None, indices_clusters=None):
        """
        the function is used for applying Time Series Word method (TSW)

        :param X: 2D ndarray
        the parameter is reduced dimensionally
        :param y: label (ignore)
        for now, it is not used.
        :return: None
        """

        self.X = X
        self.total_frame = len(self.X)
        self.n_clusters_start = len(self.X)  # The number is data number at beginning
        self.n_clusters_current = self.n_clusters_start  # The number is updated each time
        self.indices_clusters = np.arange(self.n_clusters_current)  # sequence is contained of first number of cluster

        if indices_clusters is not None:

            self.n_clusters_start = len(indices_clusters)
            self.n_clusters_current = self.n_clusters_start
            self.indices_clusters = indices_clusters

        self.__initialize()

        while self.n_clusters_current > self.n_clusters_final:
            #  the 'while' function is over when threshold (n_cluster_current) is exceeded
            if self.index_min_distance == 0:

                self.__run_head_exception()

            elif self.n_clusters_current == self.index_min_distance + 1:

                self.__run_tail_exception()

            else:

                self.__clustering_X()
                self.__compute_centroids()
                self.__compute_distances()
                self.__merge_clusters()
                self.n_clusters_current -= 1

            if self.n_clusters_current in self.list_save_n:

                self.dict_indices_clusters[self.n_clusters_current] = self.indices_clusters

            self.display_func()

        if self.display_mode == "tqdm":

            self.progress_bar.close()

        print("\nHave fitted!")

    def get_labels(self):
        #  The function is used for getting labels

        self.y = [i_clusters for i_clusters in range(self.n_clusters_final-1)
                  for i in range(self.indices_clusters[i_clusters+1] - self.indices_clusters[i_clusters])]

        self.y.extend([self.n_clusters_final-1 for i in range(self.total_frame - self.indices_clusters[self.n_clusters_final-1])])

        return self.y

    def save_indices(self, path_save, take_name):

        for n in self.dict_indices_clusters:

            file_name = take_name + "_cwo_N_indices" + "_n=" + str(n)
            np.save(path_save + file_name, self.dict_indices_clusters[n])

    def load_indices(self, path_save, take_name, n_indices_clusters):

        file_name = take_name + "_cwo_N_indices" + "_n=" + str(n_indices_clusters)

        self.indices_clusters = np.load(path_save+file_name+".npy")

        return self.indices_clusters

    def __initialize(self):

        def __initialize_X_clusters(self):

            X_clusters = []

            for i_cluster in range(self.n_clusters_start-1):
                X_clusters.append(self.X[self.indices_clusters[i_cluster]:self.indices_clusters[i_cluster+1], :])

            X_clusters.append(self.X[self.indices_clusters[-1]:, :])

            return X_clusters

        def __initialize_centroids(self):

            centroids_current = [ClassifierWardOrder_N.__compute_cluster_centroid(self.X_clusters[i_cluster])
                                 for i_cluster in range(self.n_clusters_start)]

            centroids_premerge = [ClassifierWardOrder_N.__compute_cluster_centroid(np.concatenate([self.X_clusters[i_cluster], self.X_clusters[i_cluster+1]]))
                                  for i_cluster in range(self.n_clusters_start-1)]

            return centroids_current, centroids_premerge

        def __initialize_ward_distances(self):

            sq_sum_distances_current = []
            sq_sum_distances_premerge = []

            for i_cluster in range(self.n_clusters_start):

                sum = 0

                for data in self.X_clusters[i_cluster]:

                    sum += distance.euclidean(data, self.centroids_current[i_cluster])**2

                sq_sum_distances_current.append(sum)

            for i_cluster in range(self.n_clusters_start-1):

                sum = 0

                for data in np.concatenate([self.X_clusters[i_cluster], self.X_clusters[i_cluster+1]]):

                    sum += distance.euclidean(data, self.centroids_current[i_cluster])**2

                sq_sum_distances_premerge.append(sum)

            ward_distances = [sq_sum_distances_premerge[i_cluster]
                              - (sq_sum_distances_current[i_cluster] + sq_sum_distances_current[i_cluster + 1])
                              for i_cluster in range(self.n_clusters_start-1)]

            return sq_sum_distances_current, sq_sum_distances_premerge, ward_distances

        def __merge_clusters_first(self):

            min_distance = np.min(self.ward_distances)

            index_min_distance = np.argmin(self.ward_distances)

            self.indices_clusters = np.delete(self.indices_clusters, index_min_distance + 1)

            return min_distance, index_min_distance

        print("n_clusters_start:", self.n_clusters_start)

        if self.display_mode == "tqdm":

            self.progress_bar = tqdm(total=self.n_clusters_start-self.n_clusters_final)

            print("Fitting...")

        self.X_clusters = __initialize_X_clusters(self)
        self.centroids_current, self.centroids_premerge = __initialize_centroids(self)
        self.sq_sum_distances_current, self.sq_sum_distances_premerge, self.ward_distances = __initialize_ward_distances(self)
        self.min_distance, self.index_min_distance = __merge_clusters_first(self)

        self.n_clusters_current -= 1

        if self.n_clusters_current in self.list_save_n:

            self.dict_indices_clusters[self.n_clusters_current] = self.indices_clusters

        self.display_func()

    def __clustering_X(self):

        del self.X_clusters[self.index_min_distance:self.index_min_distance+2]

        self.X_clusters.insert(self.index_min_distance, self.X[self.indices_clusters[self.index_min_distance]:self.indices_clusters[self.index_min_distance+1], :])

    def __compute_centroids(self):

        del self.centroids_current[self.index_min_distance:self.index_min_distance+2]

        self.centroids_current.insert(self.index_min_distance, self.centroids_premerge[self.index_min_distance])

        del self.centroids_premerge[self.index_min_distance-1:self.index_min_distance+2]

        self.centroids_premerge.insert(self.index_min_distance-1,
                                       ClassifierWardOrder_N.__compute_cluster_centroid(np.concatenate([self.X_clusters[self.index_min_distance-1], self.X_clusters[self.index_min_distance]])))
        self.centroids_premerge.insert(self.index_min_distance,
                                       ClassifierWardOrder_N.__compute_cluster_centroid(np.concatenate([self.X_clusters[self.index_min_distance], self.X_clusters[self.index_min_distance+1]])))

    @staticmethod
    def __compute_cluster_centroid(cluster):

        n_dimension = cluster.shape[1]

        centroid = [np.mean(cluster[:, i]) for i in range(n_dimension)]

        return centroid


    def __compute_distances(self):

        del self.sq_sum_distances_current[self.index_min_distance:self.index_min_distance+2]

        sum = 0

        for data in self.X_clusters[self.index_min_distance]:

            sum += distance.euclidean(data, self.centroids_current[self.index_min_distance])**2

        self.sq_sum_distances_current.insert(self.index_min_distance, sum)

        del self.sq_sum_distances_premerge[self.index_min_distance-1:self.index_min_distance+2]

        sum_1 = 0
        sum_2 = 0

        for data in np.concatenate([self.X_clusters[self.index_min_distance-1], self.X_clusters[self.index_min_distance]]):

            sum_1 += distance.euclidean(data, self.centroids_premerge[self.index_min_distance-1])**2

        for data in np.concatenate([self.X_clusters[self.index_min_distance], self.X_clusters[self.index_min_distance+1]]):

            sum_2 += distance.euclidean(data, self.centroids_premerge[self.index_min_distance])**2

        self.sq_sum_distances_premerge.insert(self.index_min_distance-1, sum_1)
        self.sq_sum_distances_premerge.insert(self.index_min_distance, sum_2)

        del self.ward_distances[self.index_min_distance-1:self.index_min_distance+2]

        self.ward_distances.insert(self.index_min_distance-1,
                                   self.sq_sum_distances_premerge[self.index_min_distance-1]
                                   -(self.sq_sum_distances_current[self.index_min_distance-1]+self.sq_sum_distances_current[self.index_min_distance]))
        self.ward_distances.insert(self.index_min_distance,
                                   self.sq_sum_distances_premerge[self.index_min_distance]
                                   -(self.sq_sum_distances_current[self.index_min_distance]+self.sq_sum_distances_current[self.index_min_distance+1]))

    def __merge_clusters(self):

        self.min_distance = np.min(self.ward_distances)

        self.index_min_distance = np.argmin(self.ward_distances)

        self.indices_clusters = np.delete(self.indices_clusters, self.index_min_distance+1)

    def __run_head_exception(self):

        def __clustering_X_head_ex(self):

            del self.X_clusters[0:2]

            self.X_clusters.insert(0, self.X[0:self.indices_clusters[1], :])

        def __compute_centroids_head_ex(self):

            del self.centroids_current[0:2]

            self.centroids_current.insert(0, self.centroids_premerge[0])

            del self.centroids_premerge[0:2]

            self.centroids_premerge.insert(0, ClassifierWardOrder_N.__compute_cluster_centroid(np.concatenate([self.X_clusters[0], self.X_clusters[1]])))

        def __compute_distances_head_ex(self):

            del self.sq_sum_distances_current[0:2]

            sum = 0

            for data in self.X_clusters[0]:

                sum += distance.euclidean(data, self.centroids_current[0])**2

            self.sq_sum_distances_current.insert(0, sum)

            del self.sq_sum_distances_premerge[0:2]

            sum_2 = 0

            for data in np.concatenate([self.X_clusters[0], self.X_clusters[1]]):

                sum_2 += distance.euclidean(data, self.centroids_premerge[0])**2

            self.sq_sum_distances_premerge.insert(0, sum_2)

            del self.ward_distances[0:2]

            self.ward_distances.insert(0, self.sq_sum_distances_premerge[0]
                                       - (self.sq_sum_distances_current[0] + self.sq_sum_distances_current[1]))

        __clustering_X_head_ex(self)
        __compute_centroids_head_ex(self)
        __compute_distances_head_ex(self)
        self.__merge_clusters()
        self.n_clusters_current -= 1

    def __run_tail_exception(self):

        def __clustering_X_tail_ex(self):

            del self.X_clusters[self.index_min_distance:self.index_min_distance + 2]

            self.X_clusters.insert(self.index_min_distance, self.X[self.indices_clusters[self.index_min_distance]:, :])

        def __compute_centroids_tail_ex(self):

            del self.centroids_current[self.index_min_distance:self.index_min_distance + 2]

            self.centroids_current.insert(self.index_min_distance, self.centroids_premerge[self.index_min_distance])

            del self.centroids_premerge[self.index_min_distance - 1:self.index_min_distance + 1]

            self.centroids_premerge.insert(self.index_min_distance - 1,
                                           ClassifierWardOrder_N.__compute_cluster_centroid(np.concatenate(
                                               [self.X_clusters[self.index_min_distance - 1],
                                                self.X_clusters[self.index_min_distance]])))

        def __compute_distances_tail_ex(self):

            del self.sq_sum_distances_current[self.index_min_distance:self.index_min_distance + 2]

            sum = 0

            for data in self.X_clusters[self.index_min_distance]:

                sum += distance.euclidean(data, self.centroids_current[self.index_min_distance])**2

            self.sq_sum_distances_current.insert(self.index_min_distance, sum)

            del self.sq_sum_distances_premerge[self.index_min_distance - 1:self.index_min_distance + 1]

            sum_1 = 0

            for data in np.concatenate([self.X_clusters[self.index_min_distance - 1], self.X_clusters[self.index_min_distance]]):

                sum_1 += distance.euclidean(data, self.centroids_premerge[self.index_min_distance - 1])**2

            self.sq_sum_distances_premerge.insert(self.index_min_distance - 1, sum_1)

            del self.ward_distances[self.index_min_distance - 1:self.index_min_distance + 1]

            self.ward_distances.insert(self.index_min_distance - 1,
                                       self.sq_sum_distances_premerge[self.index_min_distance - 1]
                                       - (self.sq_sum_distances_current[self.index_min_distance - 1] +
                                          self.sq_sum_distances_current[self.index_min_distance]))

        __clustering_X_tail_ex(self)
        __compute_centroids_tail_ex(self)
        __compute_distances_tail_ex(self)
        self.__merge_clusters()
        self.n_clusters_current -= 1

    def __display_normal(self):

        # print("ward_distances:", self.ward_distances)
        print("min_distance:", self.min_distance)
        print("index_min_distance:", self.index_min_distance)
        print("indices_clusters:", self.indices_clusters)
        print("n_clusters_current:", self.n_clusters_current)


    def __display_tqdm(self):

        self.progress_bar.update(1)


class ClassifierWardOrder_D(Classifier):
    """
    The user for getting data uses two function.
    firstly, you generate the class named this.
    Then, call function named "fit" for applying TSW
    Then, if you want to get labels name, you call function named "get_labels"
    """

    def __init__(self, threshold: float = 100, list_save_n=None):

        self.threshold = threshold

        if list_save_n is None:

            self.list_save_n = [0]
            self.is_save = False

        else:

            self.list_save_n = list_save_n
            self.is_save = True

        self.dict_indices_clusters = {}

    def fit(self, X, y=None):
        """
        the function is used for applying Time Series Word method (TSW)

        :param X: 2D ndarray
        the parameter is reduced dimensionally
        :param y: label (ignore)
        for now, it is not used.
        :return: None
        """

        self.X = X
        self.n_clusters_start = len(self.X)  # The number is data number at beginning
        self.n_clusters_current = self.n_clusters_start  # The number is updated each time
        self.indices_clusters = np.arange(self.n_clusters_current)  # sequence is contained of first number of cluster

        print("n_clusters_start:", self.n_clusters_start)

        while self.min_distance >= self.threshold:
            #  the 'while' function is over when threshold (n_cluster_current) is exceeded
            self.__clustering_X()
            self.__compute_centroids()
            self.__compute_distances()
            self.__merge_clusters()
            self.n_clusters_current -= 1

            if self.n_clusters_current in self.list_save_n:

                self.dict_indices_clusters[self.n_clusters_current] = self.indices_clusters

            print("n_clusters_current:", self.n_clusters_current)

        if not self.is_save:

            self.dict_indices_clusters[self.n_clusters_current] = self.indices_clusters

    def get_labels(self):
        #  The function is used for getting labels

        self.y = [i_clusters for i_clusters in range(self.n_clusters_final - 1)
                  for i in range(self.indices_clusters[i_clusters + 1] - self.indices_clusters[i_clusters])]

        self.y.extend([self.n_clusters_final - 1 for i in
                       range(self.n_clusters_start - self.indices_clusters[self.n_clusters_final - 1])])

        return self.y

    def seve_indices(self, path_save, take_name):

        for n in self.dict_indices_clusters:

            file_name = take_name + "_cwo_D_indices" + "_n=" + str(n)
            np.save(path_save+file_name, self.dict_indices_clusters[n])

    def __clustering_X(self):

        #  The function is used for initializing the list for deleting the previous list

        self.X_clusters = []

        for i_cluster in range(self.n_clusters_current - 1):
            self.X_clusters.append(self.X[self.indices_clusters[i_cluster]:self.indices_clusters[i_cluster + 1], :])

        self.X_clusters.append(self.X[self.indices_clusters[-1]:, :])

    def __compute_centroids(self):

        self.centroids_current = np.array([ClassifierWardOrder_D.__compute_cluster_centroid(self.X_clusters[i_cluster])
                                           for i_cluster in range(self.n_clusters_current)])

        self.centroids_premerge = np.array([ClassifierWardOrder_D.__compute_cluster_centroid(
            np.concatenate([self.X_clusters[i_cluster], self.X_clusters[i_cluster + 1]]))
                                            for i_cluster in range(self.n_clusters_current - 1)])

    @staticmethod
    def __compute_cluster_centroid(cluster):

        n_dimension = cluster.shape[1]

        centroid = [np.mean(cluster[:, i]) for i in range(n_dimension)]

        return centroid

    def __compute_distances(self):

        sq_sum_distances_clusters = list()
        sq_sum_distances_premerge = list()

        for i_cluster in range(self.n_clusters_current):

            sum = 0

            for data in self.X_clusters[i_cluster]:
                sum += distance.euclidean(data, self.centroids_current[i_cluster]) ** 2

            sq_sum_distances_clusters.append(sum)

        for i_cluster in range(self.n_clusters_current - 1):

            sum = 0

            for data in np.concatenate([self.X_clusters[i_cluster], self.X_clusters[i_cluster + 1]]):
                sum += distance.euclidean(data, self.centroids_premerge[i_cluster]) ** 2

            sq_sum_distances_premerge.append(sum)

        self.ward_distances = np.array([sq_sum_distances_premerge[i_cluster] - sq_sum_distances_clusters[i_cluster] -
                                        sq_sum_distances_clusters[i_cluster + 1]
                                        for i_cluster in range(self.n_clusters_current - 1)])
        print("ward_distances:", self.ward_distances)

    def __merge_clusters(self):

        self.min_distance = np.min(self.ward_distances)
        print("min_distance:", self.min_distance)

        index_min_distance = np.argmin(self.ward_distances)
        print("index_min_distance:", index_min_distance)

        self.indices_clusters = np.delete(self.indices_clusters, index_min_distance + 1)
        print("indices_clusters:", self.indices_clusters)
