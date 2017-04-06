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
        str(self)


class Segmentation(list):  # TODO
    """
    Usage:
    segm[5] - returns the segment with index 5
    segm._segments[5] - equivalent to segm[5], do not use
    segm.by_auhor[1] - returns a list of segments associated with author 1
    """

    def __init__(self, author_count, segments: List[Segment],
                 repair=False, maxRepairableError=0, document_length=-1):

        self.author_count = author_count
        # segments are sorted by offset index
        segments.sort(key=lambda x: x.offset)
        self._segments = segments

        # segments can be accessed by author index.
        self.by_author = dict((i, []) for i in range(author_count))
        for s in self:
            self.by_author[s.author].append(s)

        assert all(s.length > 0 for s in self), "Segment length must be positive."

        assert self.validate(repair, maxRepairableError), \
            "Segments must have positive lengths and not overlap."

        if document_length != -1:
            self[-1].moveEndOffsetTo(document_length)

    def validate(self, repair=False, tolerance=0) -> bool:
        # assumes segments sorted by offset, then by length
        prev = self[0]
        for i in range(1, len(self)):
            curr = self[i]
            d = curr.offset - prev.endOffset
            if d != 0:
                if repair:
                    if abs(d) > tolerance:
                        return False
                    lp, lc = prev.length, curr.length
                    ltot = max(prev.endOffset, curr.endOffset) - prev.offset
                    prev.length = max(1, ltot * lp // (lp + lc))
                    curr.length = ltot - prev.length
                    curr.offset = prev.endOffset
                else:
                    return False
            prev = curr
        return True

    @property
    def document_length(self):
        return self[-1].endOffset

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
