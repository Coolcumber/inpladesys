import numpy as np
from abc import ABC, abstractmethod
from inpladesys.datatypes import *


class AbstractBasicFeatureExtractor(ABC):  # TODO

    @abstractmethod
    def fit(self, document: Document):
        pass

    @abstractmethod
    def transform(self, documents, segments) -> np.ndarray: # TODO: define signature
        pass
