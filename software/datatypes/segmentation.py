from typing import List, Tuple


class Segment():
    def __init__(self, author, start, end):
        self.author = author
        self.start, self.end = start, end


class Segmentation():  # TODO
    def __init__(self, author_count, segments: List[Segment]):
        self.author_count = 0