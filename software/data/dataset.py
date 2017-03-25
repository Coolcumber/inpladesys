from typing import Tuple, List
from data import Document, Solution
import random


class Dataset():
    def __init__(self, documents: List[Document], solutions: List[Solution]):
        self.documents = documents
        self.solutions = solutions

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.documents[key.start:key.stop:key.step], self.solutions[key.start:key.stop:key.step]
        else:  # int
            return self.documents[key], self.solutions[key]

    @property
    def size(self) -> int:
        return len(self)

    def shuffle(self, order_determining_number: float = -1):
        """ Shuffles the data. """
        document_solution_pairs = list(zip(self.documents, self.solutions))
        if order_determining_number < 0:
            random.shuffle(document_solution_pairs)
        else:
            random.shuffle(document_solution_pairs, lambda: order_determining_number)
        self.documents[:], self.solutions[:] = zip(*document_solution_pairs)

    def split(self, start, end):
        """ Splits the dataset into two smaller datasets. """
        first = Dataset(self.documents[start:end], self.solutions[start:end])
        second = Dataset(self.documents[:start] + self.documents[end:], self.solutions[:start] + self.solutions[end:])
        return first, second

    @staticmethod
    def load(dataset_directory: str):
        documents = load_documents(dataset_directory)  # TODO
        solutions = load_solutions(dataset_directory)  # TODO
        return Dataset(documents, solutions)