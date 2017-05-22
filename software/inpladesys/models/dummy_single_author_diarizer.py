from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple
from .abstract_author_diarizer import AbstractAuthorDiarizer
from inpladesys.models.misc import fix_segmentation_labels_for_plagiarism_detection


class DummySingleAuthorDiarizer(AbstractAuthorDiarizer):  # TODO
    def train(self, dataset: Dataset=None):
        pass

    def _predict(self, document: Document) -> Segmentation:
        segm = Segmentation(2, [Segment(offset=0, length=len(document), author=0)])
        fix_segmentation_labels_for_plagiarism_detection(segm, plagiarism_majority=True)  # mark everything as plagiarism
        return segm
