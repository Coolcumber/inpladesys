from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow
from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


class StopWordsExtractor(AbstractSingleFeatureExtractor):
    def __init__(self, params=None):
        super().__init__(params)
        stop_words = stopwords.words('english')
        max_features = len(stop_words)
        self.cv = CountVectorizer(max_features=max_features,
                                  lowercase=True,
                                  vocabulary=stop_words,
                                  token_pattern='(?u)\\b\\w+\\b'
                                  )

    def fit(self, document, preprocessed_document=None, tokens=None):
        pass

    def transform(self, sliding_window: SlidingWindow):
        tokens = sliding_window.data['left_context_tokens'] + sliding_window.data['right_context_tokens']
        return self.cv.transform([" ".join(tokens)])
