import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from datatypes import *


def AbstractFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def train(self, dataset: Dataset):
        pass

    @abstractmethod
    def get_features(document, segment) -> np.ndarray:
        pass
