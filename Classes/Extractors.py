from abc import ABC, abstractmethod, abstractproperty
from pandas import DataFrame
from Classes.Interfaces import FeaturesExtractor, SamplesExtractor


class MachineFeatureExtractor(FeaturesExtractor):

    def __init__(self, metrics_info: dict = None, raw_df: DataFrame = None):

        self.metrics_info = metrics_info
        self.raw_df = raw_df

        self.extracted_df = DataFrame()

    def set_metrics(self, metrics_info: dict = None):

        self.metrics_info = metrics_info if metrics_info is not None else self.metrics_info

        self.machine_asset = self.metrics_info["machine_asset"]
        self.machine_name = self.metrics_info["machine_name"]
        self.moving_axis = self.metrics_info["moving_axis"]

    @abstractmethod
    def _extract_features(self):

        pass

    def extract(self, raw_df: DataFrame):

        self.raw_df = raw_df if raw_df is not None else self.raw_df

        self._extract_features()

        return self.extracted_df


class RigidMachineExtractor(MachineFeatureExtractor):

    def __init__(self, metrics_info: dict = None, raw_df: DataFrame = None):

        super().__init__(metrics_info=metrics_info, raw_df=raw_df)

    def set_metrics(self, metrics_info: dict = None):

        super().set_metrics(metrics_info=metrics_info)

        self.bone_info = {'flag': False,
                          'parts': [],
                          'position': ["X", "Y", "Z"],
                          'rotation': ["W", "Z", "Y", "X"]}

        self.rigid_body_info = {'flag': True,
                                'parts': [self.machine_name],
                                'position': [self.moving_axis],
                                'rotation': []}

        self.marker_info = {'flag': False,
                            'parts': [],
                            'position': ["X", "Y", "Z"],
                            'rotation': ["W", "Z", "Y", "X"]}

        self.rigid_body_marker_info = {'flag': False,
                                       'parts': [],
                                       'position': ["X", "Y", "Z"],
                                       'rotation': ["W", "Z", "Y", "X"]}





