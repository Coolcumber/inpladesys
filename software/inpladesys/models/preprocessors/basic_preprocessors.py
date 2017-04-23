
from nltk import word_tokenize
from .abstract_preprocessor import AbstractPreprocessor


class TokenizerPreprocessor(AbstractPreprocessor):

    def fit_transform(self, raw_text):
        token_spans = []
        offset = 0
        tokens = word_tokenize(raw_text)
        for token in tokens:
            offset = raw_text.find(token, offset)
            token_spans.append((token, offset, offset + len(token)))
            offset += len(token)
        return token_spans
