from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.abstract_single_feature_extractor import AbstractSingleFeatureExtractor


class BagOfWordsExtractor(AbstractSingleFeatureExtractor):

    def __init__(self, params):
        super.__init__(params)
        self.cv = CountVectorizer(max_features=params['max_features'])

    def fit(self, text_data):
        print('bow print')
        pass

    def transform(self, text_data):
        pass

p = 1
b = BagOfWordsExtractor(params=1)
