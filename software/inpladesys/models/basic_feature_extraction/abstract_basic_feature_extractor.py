import numpy as np
from abc import ABC, abstractmethod
from inpladesys.datatypes import *
import json


class AbstractBasicFeatureExtractor(ABC):  # TODO

    def __init__(self, features_file_name, preprocessor):
        self.features = json.load(open(features_file_name, 'r'))['features']
        self.preprocessor = preprocessor
        self.feature_vectors = []
        self.feature_extractors = []

    def load_class(self, module_name, class_name):
        pass

    @abstractmethod
    def fit(self, document: Document):
        raise NotImplementedError

    @abstractmethod
    def transform(self, documents, segments) -> np.ndarray: # TODO: define signature
        raise NotImplementedError
