from nltk import word_tokenize
from nltk import pos_tag
from .abstract_preprocessor import AbstractPreprocessor


class TokenizerPreprocessor(AbstractPreprocessor):
    def fit_transform(self, raw_text):
        token_spans = []
        tokens = word_tokenize(raw_text)
        pos_tags = pos_tag(tokens, tagset='universal', lang='eng')
        assert len(pos_tags) == len(tokens)
        prev_end, next_start = 0, 0
        for i in range(len(tokens)):
            token = tokens[i]
            if token == "''" or token == '``':  # http://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
                token = '"'
            try:
                next_start = raw_text.index(token, prev_end, prev_end+100) + len(token)
            except:
                a = raw_text[prev_end:prev_end + 100]
                next_start = prev_end + len(token)
                print(token)
                print(next_start)
                print(raw_text[prev_end:next_start])
                print(raw_text[next_start - 100:next_start])
                print(raw_text[next_start:next_start + 100])
                print(tokens[i - 10:i])
                print(tokens[i:i + 10])
                pass
            token_spans.append((tokens[i], prev_end, next_start, pos_tags[i][1]))  # token, start, end, pos
            prev_end = next_start
            # offset += len(token)
        return token_spans


class BasicTokenizerPreprocessor(AbstractPreprocessor):
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