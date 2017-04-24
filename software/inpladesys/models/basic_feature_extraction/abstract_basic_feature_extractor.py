import numpy as np
from abc import ABC, abstractmethod
from inpladesys.datatypes import *
import importlib
import json


class AbstractBasicFeatureExtractor(ABC):  # TODO

    def __init__(self, features_file_name):
        self.features = json.load(open(features_file_name, 'r'))['features']
        self.single_feature_extractors = []

    def load_class(self, module_name, class_name):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @abstractmethod
    def fit(self, document: Document, preprocessed_document=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, document, preprocessed_document, context_size) -> np.ndarray: # TODO: define signature
        raise NotImplementedError
