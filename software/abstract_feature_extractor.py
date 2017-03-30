import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from datatypes import *


def AbstractFeatureExtractor(ABC):  # TODO
    @abstractmethod
    def train(self, dataset: Dataset):
        pass

    def set_document(self, document):
        """
        As features of a part of a ocument might depend on the surroundong text,
        a feature extractor might like to heve the whole document available.
        """
        self.document = document

    @abstractmethod
    def get_features(self, segments) -> np.ndarray:
        """
        Returns an array of feature vectors for a list of segments or a single
        segment.
        """
        pass
