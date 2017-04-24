from abc import ABC, abstractmethod
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor


class SlidingWindow:

    def __init__(self, data: dict):
        self.data = data


class AbstractSlidingWindowIterator(ABC):

    def __init__(self, raw_document, context_size, preprocessed_document=None):
        self.preprocessed_doc = preprocessed_document
        self.raw_document = raw_document
        self.context_size = context_size

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        else:
            return self.next()

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next(self) -> SlidingWindow:
        raise NotImplementedError


class TokenBasedSlidingWindowIterator(AbstractSlidingWindowIterator):

    def __init__(self, preprocessed_document, raw_document, context_size):
        super().__init__(raw_document, context_size, preprocessed_document)
        self.token_i = 0
        self.n_tokens = len(preprocessed_document)
        self.tokens = [i[0] for i in preprocessed_document]
        self.half_context_size = context_size // 2

    def has_next(self):
        return self.token_i < self.n_tokens

    def next(self):
        token = self.tokens[self.token_i]

        left_bound = self.token_i - self.half_context_size
        left_i = left_bound if left_bound >= 0 else 0
        left_context_tokens = self.tokens[left_i:self.token_i]

        right_bound = self.token_i + self.half_context_size + 1
        right_i = right_bound if right_bound <= self.n_tokens else self.n_tokens
        right_context_tokens = self.tokens[self.token_i+1:right_i]

        start = self.preprocessed_doc[left_i][1]
        end = self.preprocessed_doc[self.token_i][1]
        raw_left_context = self.raw_document[start:end]

        start = self.preprocessed_doc[self.token_i][2]
        end = self.preprocessed_doc[right_i-1][2]
        raw_right_context = self.raw_document[start:end]

        start = self.preprocessed_doc[left_i][1]
        end = self.preprocessed_doc[right_i-1][2]
        raw_window_text = self.raw_document[start:end]

        self.token_i += 1

        data = {
            'token': token,
            'left_context_tokens': left_context_tokens,
            'right_context_tokens': right_context_tokens,
            'raw_left_context': raw_left_context,
            'raw_right_context': raw_right_context,
            'raw_window_text': raw_window_text
        }

        return SlidingWindow(data)


if False:
    raw_doc = "This is a first sentence. This is a second sentence."
    tokens = TokenizerPreprocessor().fit_transform(raw_doc)
    print(tokens)
    swi = TokenBasedSlidingWindowIterator(tokens, raw_doc, context_size=4)
    while swi.has_next():
        w = swi.next()
        d = w.data
        print('Left context:', d['left_context_tokens'])
        print('Token:', d['token'])
        print('Right context:', d['right_context_tokens'])
        print('All raw:', d['raw_window_text'])
        print('Raw left:', d['raw_left_context'])
        print('Raw right:', d['raw_right_context'])
        print()