
from nltk import word_tokenize
from .abstract_preprocessor import AbstractPreprocessor


class TokenizerPreprocessor(AbstractPreprocessor):

    def fit_transform(self, raw_text):
        token_spans = []
        offset = 0
        tokens = word_tokenize(raw_text)
        for token in tokens:
            real_token = token
            if token == "''" or token == '``':  # http://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
                 token = '"'
            offset = raw_text.index(token, offset)
            token_spans.append((real_token, offset, offset + len(token)))
            offset += len(token)
        return token_spans
