from typing import List, Tuple
from collections import namedtuple


## Factory method for creating named tuples - used similarly to a constructor
## Example: assert(Segment(17, 100, author=0).length == 100)
Segment = namedtuple('Segment', ['offset', 'length', 'author'])


class Segmentation():  # TODO
    """
    Usage:
    segm[5] - returns the segment with index 5
    segm.segments[5] - equivalent to segm[5]
    segm.by_auhor[1] - returns a list of segments associated with author 1
    """
    def __init__(self, author_count, segments: List[Segment]):
        self.author_count = author_count
        segments.sort()  # segments are sorted by offset index
        self.segments = segments
        self.by_author = dict()  # segments can be accessed by author index.
        for i in range(author_count):
            self.by_author[i] = []
        for s in segments:
            self.by_author[s.author].append(s)
        self.by_segment = segments

    def __getitem__(self, key):  # segments can be accessed by segment index.
        return self.segments[key]

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return "Segmentation(" & str(self.segments) & ")"
