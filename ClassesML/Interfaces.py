from abc import ABC, abstractmethod, abstractproperty
from typing import List
import numpy as np

class IModel(ABC):

    @abstractmethod
    def create(self):
        pass

class IDataSet(ABC):

    @abstractmethod
    def create(self):
        pass

class IKinimatics(ABC):

    @abstractmethod
    def _filtering_data(self):
        pass

    @abstractmethod
    def _label_encoding(self):
        pass

    @abstractmethod
    def _generate_distance(self):
        pass

    @abstractmethod
    def _generate_coordinate(self):
        pass

    @abstractmethod
    def _generate_rotation_matrix(self):
        pass

    @abstractmethod
    def _generate_flexion_angle(self):
        pass

    @abstractmethod
    def _generate_1st_flexion_angle(self):
        pass

    @abstractmethod
    def _generate_abduction_angle(self):
        pass

    @abstractmethod
    def _generate_kinematics(self, display):
        pass
