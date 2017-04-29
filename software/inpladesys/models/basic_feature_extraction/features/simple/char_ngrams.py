from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow
from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor


class CharNGramsExtractor(AbstractSingleFeatureExtractor):
    def __init__(self, params=None):
        super().__init__(params)
        self.cv = CountVectorizer(analyzer='char',
                                  lowercase=False,
                                  max_features=params['max_features'],
                                  ngram_range=(params['min_ngram_range'], params['max_ngram_range']),
                                  )

    def fit(self, document, preprocessed_document=None, tokens=None):
        self.cv.fit([document])

    def transform(self, sliding_window: SlidingWindow):
        # TODO use a whole raw window text ?
        return self.cv.transform([sliding_window.data['raw_left_context']]) + \
               self.cv.transform([sliding_window.data['raw_right_context']])
