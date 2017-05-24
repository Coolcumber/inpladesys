from abc import ABC
from abc import abstractmethod
from typing import List
import numpy as np
from inpladesys.datatypes import *
from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector


class AbstractDiarizer(ABC):

    @abstractmethod
    def fit_predict(self, preprocessed_documents: List[List[tuple]],
                    documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None, task=None) -> List[Segmentation]:
        """
        sklearn compatible fit_predict method. Override for model implementation.
        :param preprocessed_documents: List of lists of tuples. Each list of tuples represents
        preprocessed document obtained by preprocessor. Tuple is (token, offset, length, PoS tag)
        :param documents_features: List of 2D ndarrays. Every ndarray stores all features from a single document.
        :param dataset: Dataset. It should be used for obtaining the number of authors (if available)
        for each document and length of each document.
        :return: List of Segmentations obtained by the implemented model
        """
        raise NotImplementedError

    def get_model_selector(self) -> AbstractModelSelector:
        return None

    def get_optimal_hyperparams(self, task=None):
        return None