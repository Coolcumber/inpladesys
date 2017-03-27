from abc import ABC
from .dataset import Dataset
from .abstract_dataset_loader import AbstractDatasetLoader
import json


class Pan16DatasetLoader(AbstractDatasetLoader):
    def _init_(self, path:str):
        self.path = path

    def load(self) -> Dataset:
        documents = []
        segmentations = []
        # TODO
        return Dataset(documents, segmentations)