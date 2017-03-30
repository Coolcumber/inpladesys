from abc import ABC, abstractmethod

from data import Dataset
from datatypes import Document, Segmentation
from typing import List, Tuple


def AbstractAuthorDiarizer(ABC):  # TODO
    def train(self, dataset: Dataset):
        pass

    def predict(document: Document) -> Segmentation:
        pass
