from typing import Tuple, List
from inpladesys.datatypes import Document, Segment, Segmentation
import random


class Dataset():
    def __init__(self, documents: List[Document], segmentations: List[Segmentation]):
        self.documents = documents
        self.segmentations = segmentations
        self.rand = random.Random()

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Dataset(self.documents[key.start:key.stop:key.step],
                           self.segmentations[key.start:key.stop:key.step])
        else:  # int
            return self.documents[key], self.segmentations[key]

    @property
    def size(self) -> int:
        return len(self)

    def shuffle(self, random_state=None):
        """ Shuffles the data. """
        document_segmentation_pairs = list(
            zip(self.documents, self.segmentations))
        if random_state is not None:
            self.rand.seed(random_state)
        self.rand.shuffle(document_segmentation_pairs)
        self.documents[:], self.segmentations[:] = zip(
            *document_segmentation_pairs)

    def split(self, start, end):
        """ Splits the dataset into two smaller datasets. """
        first = Dataset(self.documents[start:end],
                        self.segmentations[start:end])
        second = Dataset(self.documents[:start] + self.documents[end:],
                         self.segmentations[:start] + self.segmentations[end:])
        return first, second
