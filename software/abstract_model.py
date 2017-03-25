from abc import ABC, abstractmethod
from data import Document, Solution
from typing import List, Tuple


def AbstractAuthorDiarizer(ABC):
    def train(self, dataset: List[Tuple[Document, Solution]]):
        pass

    def predict(document: Document) -> Solution:
        pass
