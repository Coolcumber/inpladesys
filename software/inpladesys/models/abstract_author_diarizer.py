from abc import ABC, abstractmethod
from inpladesys.datatypes import Document, Segmentation, Dataset
from typing import List
import numpy as np


class AbstractAuthorDiarizer(ABC):  # TODO
    # scikit.learn-compatible interface - either override this or train
    def fit(self, documents: List[Document], segmentations: List[Segmentation]):
        """
        Fit (or train) is used to learn a feature transformation for better 
        clustering.
        """
        self.train(Dataset(documents, segmentations))
    
    # alternative interface - either override this or fit
    def train(self, dataset: Dataset):
        self.fit(dataset.documents, dataset.segmentations)
    
    @abstractmethod
    def _predict(self, document: Document) -> Segmentation:
        pass

    def fit_predict(self, dataset: Dataset, documents_features: List[np.ndarray],
                    preprocessed_documents: List[tuple]=None) -> List[Segmentation]:
        """
        You can use this function for model selection IF IT'S APPROPRIATE.

        Don't forget to scale features if needed for an algorithm!

        :param preprocessed_documents: list of results returned by preprocessor for each document
        :param documents_features: list of ndarrays; every 2D ndarray element contains
        features for a single document (e.g. features of every sliding window in a single
        document)
        :param dataset: For using number of authors for each document, where is available
        :return: list of Segmentation elements; every element represents the segmentation
        for a single document
        """
        pass

    # Do not override if not really necessary
    def predict(self, doc) -> List[Segmentation]:
        if type(doc) is Document:
            return self._predict(doc)
        return [self._predict(d) for d in doc]


