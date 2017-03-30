from typing import List, Tuple


class Segment():
    def __init__(self, author, start, end):
        self.author = author
        self.start, self.end = start, end

    def __getitem__(self, key):
        return self.start if key == 0 else self.end

    def __lt__(self, other):
        return self.start < other.start

    def __str__(self):
        return "Segment(author={}, start={}, end={})".format(self.author, self.start, self.end)

    def length(self):
        return self.end - self.start


class Segmentation():  # TODO
    def __init__(self, author_count, segments: List[Segment]):
        self.author_count = author_count
        segments.sort()
        self.segments = segments
        self.by_author = dict()  # Segments can be accessed by author index
        for i in range(author_count):
            self.by_author[i] = []
        for s in segments:
            self.by_author[s.author].append(s)
        self.by_segment = segments

    # Segments can be accessed by segment index.
    def __getitem__(self, key):
        return self.segments[key]

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return "Segmentation(" & str(self.segments) & ")"
