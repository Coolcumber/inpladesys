import numpy as np
from abc import ABC, abstractmethod
from inpladesys.datatypes import *
import json


class AbstractBasicFeatureExtractor(ABC):  # TODO

    def __init__(self, features_file_name, preprocessor=None):
        self.features = json.load(open(features_file_name, 'r'))['features']
        self.preprocessor = preprocessor

    @abstractmethod
    def fit(self, document: Document):
        raise NotImplementedError

    @abstractmethod
    def transform(self, documents, segments) -> np.ndarray: # TODO: define signature
        raise NotImplementedError
