from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow
from inpladesys.models.basic_feature_extraction.features.abstract_single_feature_extractor import AbstractSingleFeatureExtractor
import numpy as np


class AvgTokenLengthExtractor(AbstractSingleFeatureExtractor):
    def __init__(self, params=None):
        super().__init__(params)

    def fit(self, document, preprocessed_document=None, tokens=None):
        pass

    def transform(self, sliding_window: SlidingWindow):
        # center token not included
        avg_length = 0
        count = 0
        for token in sliding_window.data['left_context_tokens']:
            avg_length += len(token)
            count += 1
        for token in sliding_window.data['right_context_tokens']:
            avg_length += len(token)
            count += 1
        return avg_length / count

