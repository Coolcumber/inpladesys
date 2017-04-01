from abc import ABC, abstractmethod
from datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple

class AbstractAuthorDiarizer(ABC):  # TODO
    # scikit.learn-compatible interface - do not override
    def fit(self, documents: List[Document], segmentations: List[Segmentation]):
        self.train(Dataset(documents, segmentations))

    @abstractmethod
    def predict(self, document: Document) -> Segmentation:
        pass

    #alternativno sučelje
    @abstractmethod
    def train(self, dataset: Dataset):
        pass