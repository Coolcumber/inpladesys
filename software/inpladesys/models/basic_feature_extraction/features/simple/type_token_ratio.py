from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class TypeTokenRatioExtractor(AbstractSingleFeatureExtractor):

    def __init__(self, params=None):
        super().__init__(params)
        self.cv = CountVectorizer(lowercase=True,
                                  ngram_range=(1, 1),
                                  binary=True,
                                  token_pattern=params['token_pattern'])

    def fit(self, document, preprocessed_document=None, tokens=None):
        self.cv.fit(tokens)

    def transform(self, sliding_window: SlidingWindow):
        data = sliding_window.data
        tokens = data['left_context_tokens'] + data['right_context_tokens']
        return len(self.cv.vocabulary_) / len(tokens)

