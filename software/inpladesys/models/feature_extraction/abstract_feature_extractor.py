import numpy as np
from abc import ABC, abstractmethod
from typing import List


class AbstractFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def fit(self, X: List[np.ndarray], G: List[np.ndarray]):
        # with self.graph.as_default():
        """
        :param X: list of 2D arrays (or lists) containing vectors as rows
        :param G: list of arrays (or lists) of integers representing groups
        """
        pass

    @abstractmethod
    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
         :param X: list of 2D arrays (or lists) containing vectors as rows
        """
        pass
