
from nltk import word_tokenize
from .abstract_preprocessor import AbstractPreprocessor


class TokenizerPreprocessor(AbstractPreprocessor):

    def fit_transform(self, raw_text_data):
        return word_tokenize(raw_text_data)
