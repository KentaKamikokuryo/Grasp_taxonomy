import numpy as np
from numpy import ndarray
from typing import List, Dict
from sklearn import decomposition
from sklearn import manifold
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from abc import ABC, ABCMeta, abstractmethod, abstractproperty


class IModelDimensionalityReduction(ABC):

    @abstractmethod
    def __init__(self, hyper_dict: Dict):
        pass

    @abstractmethod
    def create(self):
        pass


class PCA(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):

        self.n_components = hyper_dict["n_components"]

    def create(self):

        model = decomposition.PCA(n_components=self.n_components)

        return model


class RPCA(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):

        self.cumulative_contribution_ratio = hyper_dict["cumulative_contribution_ratio"]

    def create(self):

        model = RatioPCA(cumulative_contribution_ratio=self.cumulative_contribution_ratio)

        return model


class LLE(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):

        self.n_neighbors = hyper_dict["n_neighbors"]
        self.n_components = hyper_dict["n_components"]

    def create(self):

        model = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components,
                                       method='modified', eigen_solver='dense')

        return model


class ISOMAP(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):

        self.n_neighbors = hyper_dict["n_neighbors"]
        self.n_components = hyper_dict["n_components"]

    def create(self):

        model = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components)

        return model


class TSNE(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):

        self.perplexity = hyper_dict["perplexity"]
        self.learning_rate = hyper_dict["learning_rate"]
        self.n_components = hyper_dict["n_components"]

    def create(self):

        model = manifold.TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_components=self.n_components,
                              random_state=0, n_iter=300, verbose=1)

        return model


class NONE(IModelDimensionalityReduction):

    def __init__(self, hyper_dict: Dict):
        pass

    def create(self):

        model = None

        return model


class RatioPCA():

    def __init__(self, cumulative_contribution_ratio: float = 0.99):

        self.cumulative_contribution_ratio = cumulative_contribution_ratio

        self.pca_model = decomposition.PCA()

    def _compute_n_components(self):

        self.n_components = 0
        self.cumulative_contribution_ratio_counter = 0

        for ratio in self.pca_model.explained_variance_ratio_:

            self.n_components += 1
            self.cumulative_contribution_ratio_counter += ratio

            if self.cumulative_contribution_ratio_counter >= 0.99:

                print('n_components =', self.n_components)
                print('cumulative_contribution_ratio_counter =', self.cumulative_contribution_ratio_counter)

                break

    def fit(self, X: ndarray, y: ndarray = None):

        self.pca_model.n_components = X.shape[1]

        self.pca_model.fit(X=X, y=y)

        self._compute_n_components()

        return self

    def fit_transform(self, X: ndarray, y = None):

        self.pca_model.n_components = X.shape[1]

        Z = self.pca_model.fit_transform(X=X, y=y)

        self._compute_n_components()
        Z = Z[:, :self.n_components]

        return Z

    def transform(self, X: ndarray):

        Z = self.pca_model.transform(X=X)

        Z = Z[:, :self.n_components]

        return Z


class ModelDimensionalityReductionFactory():

    model: IModelDimensionalityReduction

    def __init__(self, hyper_dict: Dict):

        self.hyper_dict = hyper_dict
        self.method_name = hyper_dict["method_name"]

    def create_model(self):

        if self.method_name == "PCA":
            self.model = PCA(hyper_dict=self.hyper_dict)

        elif self.method_name == "RPCA":
            self.model = RPCA(hyper_dict=self.hyper_dict)

        elif self.method_name == "LLE":
            self.model = LLE(hyper_dict=self.hyper_dict)

        elif self.method_name == "ISOMAP":
            self.model = ISOMAP(hyper_dict=self.hyper_dict)

        elif self.method_name == "TSNE":
            self.model = TSNE(hyper_dict=self.hyper_dict)

        else:
            self.model = NONE(hyper_dict=self.hyper_dict)

        sk_model = self.model.create()

        return sk_model


class ModelDimensionalityReduction():

    def __init__(self, hyper_dict: Dict,
                 x_fit: ndarray = None, y_fit: ndarray = None,
                 x_transform: ndarray = None, y_transform: ndarray = None):

        self.hyper_dict = hyper_dict
        self.method_name = self.hyper_dict["method_name"]
        self.factory = ModelDimensionalityReductionFactory(hyper_dict=self.hyper_dict)

        self.x_fit = x_fit
        self.y_fit = y_fit

        self.x_transform = x_transform
        self.y_transform = y_transform

    def create(self):

        self.model = self.factory.create_model()

    def fit(self, x_fit: ndarray = None, y_fit: ndarray = None):

        self.x_fit = x_fit if x_fit is not None else self.x_fit
        self.y_fit = y_fit if y_fit is not None else self.y_fit

        self.model.fit(self.x_fit, self.y_fit)

        self.z_fit = self.model.transform(self.x_fit)

        return self.z_fit

    def transform(self, x_transform: ndarray = None, y_transform: ndarray = None):

        self.x_transform = x_transform if x_transform is not None else self.x_transform
        self.y_transform = y_transform if y_transform is not None else self.y_transform

        self.z_transform = self.model.transform(self.x_transform)

    def fit_transform(self, x_fit: ndarray = None, y_fit: ndarray = None, x_transform: ndarray = None, y_transform: ndarray = None):

        self.x_fit = x_fit if x_fit is not None else self.x_fit
        self.y_fit = y_fit if y_fit is not None else self.y_fit

        self.x_transform = x_transform if x_transform is not None else self.x_transform
        self.y_transform = y_transform if y_transform is not None else self.y_transform

        border_index = x_fit.shape[0]

        x_all = np.concatenate([x_fit, x_transform], axis=0)
        y_all = np.concatenate([y_fit, y_transform], axis=0)

        z_all = self.model.fit_transform(x_all, y_all)

        z_fit = z_all[:border_index]
        z_transform = z_all[border_index:]

        return z_fit, z_transform, z_all
