from abc import ABC, abstractmethod
from inpladesys.models.basic_feature_extraction.sliding_window import SlidingWindow


class AbstractSingleFeatureExtractor(ABC):

    def __init__(self, params=None):
        self.params = params

    @abstractmethod
    def fit(self, document, preprocessed_document=None):  #TODO is preprocessed_document even needed here ??
        pass

    @abstractmethod
    def transform(self, sliding_window: SlidingWindow):
        pass