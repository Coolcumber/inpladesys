from typing import List, Tuple
from collections import namedtuple


# Factory method for creating named tuples - used similarly to a constructor
# Example: assert(Segment(17, 100, author=0).length == 100)
Segment = namedtuple('Segment', ['offset', 'length', 'author'])


class Segmentation(list):  # TODO
    """
    Usage:
    segm[5] - returns the segment with index 5
    segm._segments[5] - equivalent to segm[5], do not use
    segm.by_auhor[1] - returns a list of segments associated with author 1
    """

    def __init__(self, author_count, segments: List[Segment]):
        self.author_count = author_count
        segments.sort()  # segments are sorted by offset index
        self._segments = segments
        self.by_author = dict()  # segments can be accessed by author index.
        self.segment_length_sum = sum(s.length for s in segments)
        for i in range(author_count):
            self.by_author[i] = []
        for s in segments:
            self.by_author[s.author].append(s)
        self.by_segment = segments
        assert self.is_valid(), "Segments must have positive lengths and not overlap."

    def is_valid(self) -> bool:
        prev = self[0]
        for i in range(1, len(self)):
            if prev.length == 0 or self[i].offset != prev.offset + prev.length:
                return False
            prev = self[i]
        if prev.length == 0:
            return False
        return True

    # segments can be accessed by index.
    def __getitem__(self, key) -> Segment:
        return self._segments[key]

    def __len__(self):
        return len(self._segments)

    def __str__(self):
        return "Segmentation(" + str(self._segments) + ")"

    def __iter__(self):
        return self._segments.__iter__()

    def to_char_sequence(self, length_factor=1):
        return ''.join([chr(ord('0') + s.author) * int(s.length * length_factor + 0.5) for s in self._segments])
