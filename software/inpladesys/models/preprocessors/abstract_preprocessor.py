from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):

    @abstractmethod
    def fit_transform(self, raw_text_data):
        pass
