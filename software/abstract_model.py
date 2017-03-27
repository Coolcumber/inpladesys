from abc import ABC, abstractmethod
from datatypes import Document, Segmentation
from data import Dataset
from typing import List, Tuple


def AbstractAuthorDiarizer(ABC):  # TODO
    def train(self, dataset: Dataset):
        pass

    def predict(document: Document) -> Segmentation:
        pass
