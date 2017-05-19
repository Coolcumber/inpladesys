from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class CharNGramsOfPOSTagsExtractor(AbstractSingleFeatureExtractor):

    def __init__(self, params=None):
        super().__init__(params)
        self.cv = CountVectorizer(analyzer='char',
                                  lowercase=False,
                                  #max_features=params['max_features'],  # TODO should we use max_features arg ?
                                  ngram_range=(params['min_ngram_range'], params['max_ngram_range']),
                                  )

    def fit(self, document, preprocessed_document=None, tokens=None):
        pos_document = " ".join([t[3] for t in preprocessed_document])
        self.cv.fit([pos_document])

    def transform(self, sliding_window: SlidingWindow):
        data = sliding_window.data
        return self.cv.transform([" ".join(data['left_context_pos_tags'])]) + \
               self.cv.transform([" ".join(data['right_context_pos_tags'])])

