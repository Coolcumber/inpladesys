from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple

from inpladesys.models.misc import fix_segmentation_labels_for_plagiarism_detection
from .abstract_author_diarizer import AbstractAuthorDiarizer
import numpy as np
import random


class DummyStochasticAuthorDiarizer(AbstractAuthorDiarizer):
    def __init__(self, author_count=5, average_segment_length=500, random_state=-1):
        self.random_state = random_state
        self.n = author_count
        self.asl = average_segment_length
        self.probs = None
        self.rand = random.Random()

    def train(self, dataset: Dataset):
        def get_author_distribution(segmtn):
            lens = np.array([sum(s.length for s in segms)
                             for segms in segmtn.by_author.values()])
            if len(lens) < self.n:
                z = np.zeros(self.n)
                z[:len(lens)] = lens
                lens = z
            lens.sort()
            lens = lens[::-1][:self.n] / sum(lens)
            return lens
        self.probs = np.average([get_author_distribution(segmtn)
                                 for segmtn in dataset.segmentations], axis=0)

    def _predict(self, document: Document) -> Segmentation:
        def select_author():
            r = self.rand.random()
            for i, p in enumerate(np.cumsum(self.probs)):
                if r < p:
                    return i
            return 0  # never reached
        self.rand.seed(self.random_state)
        segments = []
        author = select_author()
        offset = 0
        for i in range(len(document)):
            if self.rand.random() < (1 - self.probs[author]) / self.asl:
                segments.append(
                    Segment(offset=offset, length=i - offset, author=author))
                prev = author
                while author == prev:
                    author = select_author()
                offset = i
        segments.append(
            Segment(offset=offset, length=len(document) - offset, author=author))
        segm = Segmentation(self.n, segments)
        if(segm.author_count==2):
            fix_segmentation_labels_for_plagiarism_detection(segm, plagiarism_majority=True)
        return segm
