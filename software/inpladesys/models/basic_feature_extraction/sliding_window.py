from abc import ABC
from nltk import word_tokenize


class AbstractSlidingWindow(ABC):  # TODO is this ok ??

    def get_data(self):
        return NotImplementedError


class TokenBasedSlidingWindow(AbstractSlidingWindow):

    def __init__(self, token, left_context, right_context, raw_window_text):
        self.token = token
        self.left_context = left_context
        self.right_context = right_context
        self.raw_window_text = raw_window_text

    def get_data(self):
        return {'token': self.token,
                'left_context': self.left_context,
                'right_context': self.right_context,
                'raw_window_text': self.raw_window_text}


class TokenBasedSlidingWindowIterator:

    def __init__(self, document_tokens, raw_doc, half_context_size):
        self.token_i = 0
        self.last_left_raw_i = 0
        self.n_tokens = len(document_tokens)
        self.tokens = document_tokens
        self.raw_doc = raw_doc
        self.half_context_size = half_context_size

    def __iter__(self):
        return self

    def has_next(self):
        return self.token_i < self.n_tokens

    def next(self):
        token = self.tokens[self.token_i]

        left_bound = self.token_i - self.half_context_size
        left_i = left_bound if left_bound >= 0 else 0
        left_context = self.tokens[left_i:self.token_i]

        right_bound = self.token_i + self.half_context_size + 1
        right_i = right_bound if right_bound <= self.n_tokens else self.n_tokens
        right_context = self.tokens[self.token_i+1:right_i]

        self.token_i += 1

        left_raw_i, middle_i, right_raw_i = 0, 0, 0

        if len(left_context) == 0:
            left_raw_i = 0
        else:
            left_raw_i = self.raw_doc.index(left_context[0], self.last_left_raw_i)
            self.last_left_raw_i = left_raw_i #+= len(left_context[0])

        middle_i = left_raw_i
        while middle_i <= left_raw_i != 0:
            middle_i = self.raw_doc.index(token, left_raw_i)  # TODO doesn't work for dots

        if len(right_context) == 0:
            right_raw_i = len(self.raw_doc)
        else:
            right_raw_i = middle_i
            while right_raw_i <= middle_i:
                right_raw_i = self.raw_doc.index(right_context[-1], middle_i)
            right_raw_i += len(right_context[-1])

        raw_window_text = self.raw_doc[left_raw_i:right_raw_i]

        return TokenBasedSlidingWindow(token, left_context, right_context, raw_window_text)


if True:
    raw_doc = "This is a first sentence. This is a second sentence."
    tokens = word_tokenize(raw_doc)
    print(tokens)
    swi = TokenBasedSlidingWindowIterator(tokens, raw_doc, half_context_size=2)
    while swi.has_next():
        w = swi.next()
        d = w.get_data()
        print('Left context:', d['left_context'])
        print('Token:', d['token'])
        print('Right context:', d['right_context'])
        print('Raw:', d['raw_window_text'])
        print()











