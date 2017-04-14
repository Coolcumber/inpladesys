from collections import namedtuple

from typing import List, Tuple


# Factory method for creating named tuples - used similarly to a constructor
# Example: assert(Segment(17, 100, author=0).length == 100)

class Segment():  # namedtuple('Segment', ['offset', 'length', 'author'])):
    """
    endOffset is exclusive, i.e. djacent segments (s1 s2) should have 
    s1.endOffset = s2.offset.
    """
    """def __new__(cls, offset, length, author):
        self = super(Segment, cls).__new__(cls, offset, length, author)
        # self.end = offset + length
        return self"""

    def __init__(self, offset, length, author):
        self.offset, self.length, self.author = offset, length, author

    @property
    def endOffset(self):
        return self.offset + self.length

    def moveEndOffsetTo(self, value):
        self.length += value - self.offset

    def moveOffsetTo(self, value):
        self.length += value - self.offset
        self.offset = value

    def coversCompletely(self, other):
        return self.offset <= other.offset and other.endOffset <= self.endOffset

    def coversBeginningOf(self, other):
        return self.offset <= other.offset and self.endOffset < other.endOffset

    def __str__(self):
        return "Segment(offset={}, length={}, author={})".format(self.offset, self.length, self.author)

    def __repr__(self):
        return "Segment(offset={}, length={}, author={})".format(self.offset, self.length, self.author)


class Segmentation(list):  # TODO
    """
    Usage:
    segm[5] - returns the segment with index 5
    segm._segments[5] - equivalent to segm[5], do not use
    segm.by_auhor[1] - returns a list of segments associated with author 1
    """

    def __init__(self, author_count, segments: List[Segment],
                 maxRepairableError=0, document_length=-1):
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
                " Segment {} has length {}.".format(i, s.length)

        possible_error = self.fix_if_possible(maxRepairableError)
        assert possible_error is None, \
            "Consecutive segments must not overlap or be disjoint" + \
            " by more than maxRepairableError={}. Error at segment {}."\
            .format(maxRepairableError, possible_error) + \
            " Context (the segment and its neighbours): " + str(self[possible_error - 1: possible_error + 2])

        if document_length != -1:
            assert(abs(document_length - 1 -
                       self[-1].endOffset) < maxRepairableError)
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

    def __str__(self):
        return "Segmentation(" + str(self) + ")"

    def to_char_sequence(self, length_factor=1):
        return ''.join([chr(ord('0') + s.author) * int(s.length * length_factor + 0.5) for s in self])
