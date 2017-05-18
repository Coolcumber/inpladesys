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

    def contains(self, n: int):
        return n >= self.offset and n < self.endOffset

    def __str__(self):
        return "Segment(offset={}, length={}, author={})".format(self.offset,
                                                                 self.length,
                                                                 self.author)

    def __repr__(self):
        return "Segment(offset={}, length={}, author={})".format(self.offset,
                                                                 self.length,
                                                                 self.author)
