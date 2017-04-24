from sklearn.feature_extraction.text import CountVectorizer

from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class BagOfWordsExtractor(AbstractSingleFeatureExtractor):

    def __init__(self, params):
        super().__init__(params)
        self.cv = CountVectorizer(max_features=params['max_features'])

    def fit(self, document, preprocessed_document=None):  #TODO
        print('BagOfWordsExtractor:', self.params)
        pass

    def transform(self, sliding_window: SlidingWindow):
        #print('BagOfWordsExtractor: transform')
        return [1,2,3]  # TODO implement
