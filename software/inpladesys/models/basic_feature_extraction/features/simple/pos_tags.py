from sklearn.feature_extraction.text import CountVectorizer
from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow
from nltk import pos_tag


class POSTagCountExtractor(AbstractSingleFeatureExtractor):
    def __init__(self, params=None):
        super().__init__(params)

        univ_tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN',
                       'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']

        self.cv = CountVectorizer(lowercase=False,
                                  vocabulary=univ_tagset,
                                  token_pattern='(\\b[A-Z]+\\b|\\.)')

    def fit(self, document, preprocessed_document=None, tokens=None):
        pass

    def transform(self, sliding_window: SlidingWindow):
        pos_tags = sliding_window.data['left_context_pos_tags'] + sliding_window.data['right_context_pos_tags']
        return self.cv.transform([" ".join(pos_tags)])
