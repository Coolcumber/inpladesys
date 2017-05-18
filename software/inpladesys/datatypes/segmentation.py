from collections import namedtuple
from typing import List, Tuple
from .segment import Segment
import numpy as np


class Segmentation(list):  # TODO
    """
    Usage:
    segm[5] - returns the segment with index 5
    segm._segments[5] - equivalent to segm[5], do not use
    segm.by_auhor[1] - returns a list of segments associated with author 1
    """

    def __init__(self, author_count, segments: List[Segment],
                 max_repairable_error=0, document_length=-1):
        # segments are sorted by offset index
        self.extend(segments)
        self.sort(key=lambda x: x.offset)

        # segments can be accessed by author index.
        self.author_count = author_count
        self.by_author = dict((i, []) for i in range(author_count))
        for s in self:
            self.by_author[s.author].append(s)

        for i, s in enumerate(self):
            assert s.length > 0, "Segment length must be positive." + \
                                 " Segment {} has length {}.".format(i,
                                                                     s.length)

        possible_error = self.fix_if_possible(max_repairable_error)
        assert possible_error is None, \
            "Consecutive segments must not overlap or be disjoint" + \
            " by more than max_repairable_error={}. Error at segment {}." \
                .format(max_repairable_error, possible_error) + \
            " Context (the segment and its neighbours): " + str(
                self[possible_error - 1: possible_error + 2])

        if document_length != -1:
            assert (abs(document_length - 1 -
                        self[-1].endOffset) < max_repairable_error)
            self[-1].moveEndOffsetTo(document_length - 1)

    def fix_if_possible(self, tolerance=0) -> bool:
        # assumes segments sorted by offset, then by length
        prev = self[0]
        for i in range(1, len(self)):
            curr = self[i]
            d = curr.offset - prev.endOffset
            if d != 0:
                if tolerance > 0:
                    if abs(d) > tolerance:
                        return i
                    lp, lc = prev.length, curr.length
                    ltot = max(prev.endOffset, curr.endOffset) - prev.offset
                    prev.length = max(1, ltot * lp // (lp + lc))
                    curr.length = ltot - prev.length
                    curr.offset = prev.endOffset
                else:
                    return i
            prev = curr
        return None

    @property
    def document_length(self):
        return self[-1].endOffset

    def offsets_to_authors(self, offsets):  # TODO optimize
        authors = []
        j = 0
        for i, p in enumerate(offsets):
            while not self[j].contains(p):
                j += 1
            authors.append(self[j].author)
        return np.array(authors)

    def __str__(self):
        return "Segmentation(" + str(self) + ")"

    def to_char_sequence(self, length_factor=1):
        return ''.join(
            [chr(ord('0') + s.author) * int(s.length * length_factor + 0.5) for
             s in self])
