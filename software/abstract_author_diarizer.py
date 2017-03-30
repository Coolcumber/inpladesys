from abc import ABC, abstractmethod
from datatypes import *
from typing import List, Tuple


def AbstractAuthorDiarizer(ABC):  # TODO
    @abstractmethod
    def train(self, dataset: Dataset):
        pass

    @abstractmethod
    def predict(document: Document) -> Segmentation:
        pass
