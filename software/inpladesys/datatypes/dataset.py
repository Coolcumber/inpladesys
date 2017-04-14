from typing import Tuple, List
from inpladesys.datatypes import Document, Segment, Segmentation
import random


class Dataset():
    def __init__(self, documents: List[Document], segmentations: List[Segmentation]):
        self.documents = documents
        self.segmentations = segmentations

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return (self.documents[key.start:key.stop:key.step],
                    self.segmentations[key.start:key.stop:key.step])
        else:  # int
            return self.documents[key], self.segmentations[key]

    @property
    def size(self) -> int:
        return len(self)

    def shuffle(self, order_determining_number: float = -1):
        """ Shuffles the data. """
        document_segmentation_pairs = list(
            zip(self.documents, self.segmentations))
        if order_determining_number < 0:
            random.shuffle(document_segmentation_pairs)
        else:
            random.shuffle(document_segmentation_pairs,
                           lambda: order_determining_number)
        self.documents[:], self.segmentations[:] = zip(
            *document_segmentation_pairs)

    def split(self, start, end):
        """ Splits the dataset into two smaller datasets. """
        first = Dataset(self.documents[start:end],
                        self.segmentations[start:end])
        second = Dataset(self.documents[:start] + self.documents[end:],
                         self.segmentations[:start] + self.segmentations[end:])
        return first, second
