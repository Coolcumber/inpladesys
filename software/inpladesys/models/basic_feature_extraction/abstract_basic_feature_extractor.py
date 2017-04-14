import numpy as np
from abc import ABC, abstractmethod
from inpladesys.datatypes import *


class AbstractBasicFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def fit(self, dataset: Dataset):  # TODO: define signature
        pass

    @abstractmethod
    def predict(self, documents, segments) -> np.ndarray: # TODO: define signature
        pass
