from abc import ABC, abstractmethod
from datatypes import Document, Segmentation, Dataset
from typing import List


class AbstractDatasetLoader(ABC):
    def __init__(self, dataset_dir: str):
        pass
        
    def load_dataset(self) -> Dataset:
        return Dataset(self.load_documents(), self.load_segmentations())

    @abstractmethod
    def load_documents(self) -> List[str]:
        pass

    @abstractmethod
    def load_segmentations(self) -> List[Segmentation]:
        pass
