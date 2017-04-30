
from nltk import word_tokenize
from nltk import pos_tag
from .abstract_preprocessor import AbstractPreprocessor


class TokenizerPreprocessor(AbstractPreprocessor):

    def fit_transform(self, raw_text):
        token_spans = []
        offset = 0
        tokens = word_tokenize(raw_text)
        pos_tags = pos_tag(tokens, tagset='universal', lang='eng')
        assert len(pos_tags) == len(tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            real_token = tokens[i]
            if token == "''" or token == '``':  # http://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
                 token = '"'
            offset = raw_text.index(token, offset)
            token_spans.append((real_token, offset, offset + len(token), pos_tags[i][1]))
            offset += len(token)
        return token_spans
