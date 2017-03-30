from typing import List, Tuple
from collections import namedtuple


Segment = namedtuple('Segment', ['offset', 'length', 'author'])


class Segmentation():  # TODO
    def __init__(self, author_count, segments: List[Segment]):
        self.author_count = author_count
        segments.sort()
        self.segments = segments
        self.by_author = dict()  # Segments can be accessed by author index.
        for i in range(author_count):
            self.by_author[i] = []
        for s in segments:
            self.by_author[s.author].append(s)
        self.by_segment = segments

    def __getitem__(self, key):  # Segments can be accessed by segment index.
        return self.segments[key]

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return "Segmentation(" & str(self.segments) & ")"
