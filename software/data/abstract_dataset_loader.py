from abc import ABC, abstractmethod
from data import Dataset
from datatypes import Document, Segmentation
from typing import List


class AbstractDatasetLoader(ABC):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def load_dataset(self) -> Dataset:
        documents = self.load_documents()
        segmentations = self.load_segmentations()
        return Dataset(documents, segmentations)

    @abstractmethod
    def load_documents(self) -> List[str]:
        pass

    @abstractmethod
    def load_segmentations(self) -> List[Segmentation]:
        pass
