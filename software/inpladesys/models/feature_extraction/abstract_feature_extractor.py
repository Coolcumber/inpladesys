import numpy as np
from abc import ABC, abstractmethod


class AbstractFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array of feature vectors for a list of segments or a single
        segment.
        """
        pass
