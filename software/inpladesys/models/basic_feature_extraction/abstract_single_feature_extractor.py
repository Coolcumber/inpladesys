from abc import ABC, abstractmethod
import json


class AbstractSingleFeatureExtractor(ABC):

    def __init__(self, params=None):
        self.params = params

    @abstractmethod
    def fit(self, document, preprocessed_document=None):  # TODO should be just transform ? define signature...
        pass

    @abstractmethod
    def transform(self, sliding_window: SlidingWindow):
        pass