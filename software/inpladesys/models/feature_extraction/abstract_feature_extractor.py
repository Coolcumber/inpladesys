import numpy as np
from abc import ABC, abstractmethod
from typing import List


class AbstractFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def fit(self, X: List[np.ndarray], G: List[np.ndarray]):
        # with self.graph.as_default():
        """
        :param X: list of 2D arrays (or lists) containing feature vectors
        :param G: list of arrays (or lists) of integers representing groups 
        for each document
        """
        pass

    @abstractmethod
    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
         :param X: list of 2D arrays (or lists) containing feature vectors
        """
        pass

    @abstractmethod
    def _transform_one(self, X: np.ndarray) -> Listnp.ndarray:
        """
         :param X: list of 2D arrays (or lists) containing feature vectors
        """
        pass
