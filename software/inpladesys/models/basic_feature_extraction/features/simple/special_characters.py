import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class SpecialCharactersExtractor(AbstractSingleFeatureExtractor):
    """
    Spaces are currently not included.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.cv = CountVectorizer(max_features=params['max_features'],
                                  token_pattern=params['token_pattern'])

    def fit(self, document, preprocessed_document=None, tokens=None):
        self.cv.fit(tokens)

    def transform(self, sliding_window: SlidingWindow):
        return np.sum(self.cv.transform(sliding_window.data['left_context_tokens']).toarray(), axis=0) + \
               np.sum(self.cv.transform(sliding_window.data['right_context_tokens']).toarray(), axis=0)

