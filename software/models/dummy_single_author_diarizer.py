from datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple
from .abstract_author_diarizer import AbstractAuthorDiarizer


class DummySingleAuthorDiarizer(AbstractAuthorDiarizer):  # TODO
    def train(self, dataset: Dataset):
        pass

    def _predict(self, document: Document) -> Segmentation:
        return Segmentation(1, [Segment(offset=0, length=len(document), author=0)])
