from .abstract_basic_feature_extractor import AbstractBasicFeatureExtractor
from inpladesys.datatypes import Document
import numpy as np


class BasicFeatureExtractor(AbstractBasicFeatureExtractor):

    def fit(self, document: Document):
        pass
        # if self.preprocessor
        # preprocess
        # iterate through sliding windows

    def transform(self, documents, segments) -> np.ndarray:
        return super().transform(documents, segments)