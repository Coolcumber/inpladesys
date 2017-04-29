from sklearn.feature_extraction.text import CountVectorizer

from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class BagOfWordsExtractor(AbstractSingleFeatureExtractor):

    def __init__(self, params):
        super().__init__(params)
        self.cv = CountVectorizer(max_features=params['max_features'],
                                  stop_words='english',  # is it better to use list from nltk ?
                                  lowercase=bool(params['lowercase']),  # does this makes sense: Convert all characters to lowercase before tokenizing.
                                  ngram_range=(params['min_ngram_range'], params['max_ngram_range']),
                                  token_pattern=params['token_pattern'])  # default is "(?u)\\b\\w+\\b" (tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator))

    def fit(self, document, preprocessed_document=None, tokens=None):
        self.cv.fit(tokens)

    def transform(self, sliding_window: SlidingWindow):
        data = sliding_window.data
        tokens = data['left_context_tokens'] + data['right_context_tokens']
        return self.cv.transform([" ".join(tokens)])  # TODO use raw contexts for better performance ?
