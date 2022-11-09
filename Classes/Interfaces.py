from abc import ABC, abstractmethod, abstractproperty
from pandas import DataFrame
import pandas as pd


class Extractor(ABC):

    @abstractmethod
    def __init__(self, metrics_info: dict, raw_df: DataFrame):
        pass

    @abstractmethod
    def set_metrics(self, metrics_info: dict):
        pass

    @abstractmethod
    def extract(self, raw_df: DataFrame):
        pass


class FeaturesExtractor(Extractor):

    @abstractmethod
    def __init__(self, metrics_info: dict, raw_df: DataFrame):
        pass

    @abstractmethod
    def set_metrics(self, metrics_info: dict):
        pass

    @abstractmethod
    def _extract_features(self):
        pass

    @abstractmethod
    def extract(self, raw_df: DataFrame):
        pass


class SamplesExtractor(Extractor):

    @abstractmethod
    def __init__(self, metrics_info: dict, raw_df: DataFrame):
        pass

    @abstractmethod
    def set_metrics(self, metrics_info: dict):
        pass

    @abstractmethod
    def _extract_metric_feature(self):
        pass

    @abstractmethod
    def _extract_samples(self):
        pass

    @abstractmethod
    def extract(self, raw_df: DataFrame):
        pass
