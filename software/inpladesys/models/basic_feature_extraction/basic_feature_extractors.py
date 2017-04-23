from .abstract_basic_feature_extractor import AbstractBasicFeatureExtractor
from inpladesys.datatypes import Document
import numpy as np
import importlib


class BasicFeatureExtractor(AbstractBasicFeatureExtractor):

    def fit(self, document: Document):
        preprocessed_document = None
        if self.preprocessor is not None:
            preprocessed_document = self.preprocessor.fit_transform(document)

        for feature in self.features:
            if feature['used'] == 1:
                class_name = feature['class_name']
                module_name = feature['module_name']
                params = feature['params']
                module = importlib.import_module(module_name)
                FeatureExtractorClass = getattr(module, class_name)
                feature_extractor = FeatureExtractorClass(params)
                feature_extractor.fit(document, preprocessed_document)
                self.feature_extractors.append(feature_extractor)

    def transform(self, documents, segments) -> np.ndarray:
        return super().transform(documents, segments)