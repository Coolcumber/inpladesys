from abc import ABC
from .dataset import Dataset


class AbstractDatasetLoader(ABC):
    def _init_(self, path:str):
        pass

    def load(self) -> Dataset:
        pass