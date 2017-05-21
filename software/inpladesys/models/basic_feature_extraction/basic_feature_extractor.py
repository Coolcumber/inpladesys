from inpladesys.models.basic_feature_extraction.abstract_basic_feature_extractor import AbstractBasicFeatureExtractor
from inpladesys.datatypes import Document
from inpladesys.models.basic_feature_extraction.sliding_window import TokenBasedSlidingWindowIterator
from scipy import sparse
import numpy as np


class BasicFeatureExtractor(AbstractBasicFeatureExtractor):
    def __init__(self, features_file_name, context_size=50):
        super().__init__(features_file_name)
        self.context_size = context_size

    def fit(self, document: Document, preprocessed_document=None):
        tokens = [t[0] for t in preprocessed_document]
        # single feature extractor objects should be deleted before every fit
        del self.single_feature_extractors[:]
        for feature in self.features:
            if feature['used'] == 1:
                class_name = feature['class_name']
                module_name = feature['module_name']
                params = feature['params']
                FeatureExtractorClass = self.load_class(
                    module_name, class_name)
                feature_extractor = FeatureExtractorClass(params)
                feature_extractor.fit(document, preprocessed_document, tokens)
                self.single_feature_extractors.append(feature_extractor)

    def transform(self, document, preprocessed_document, context_size=None, use_sparse=False) -> np.ndarray:
        if context_size is None:
            context_size = self.context_size
        swi = TokenBasedSlidingWindowIterator(
            preprocessed_document, document, context_size)
        feature_vectors = []
        while swi.has_next():
            sliding_window = swi.next()
            feature_vector = []
            for feature_extractor in self.single_feature_extractors:
                feature_vector.append(
                    feature_extractor.transform(sliding_window))
            # print(feature_vector)
            feature_vector = sparse.hstack(
                feature_vector, dtype=np.float32).toarray()  # TODO leave data sparse ?
            feature_vectors.append(feature_vector)
        array = np.array(feature_vectors).reshape((len(feature_vectors), feature_vector.shape[1]))
        if use_sparse:
            return sparse.csr_matrix(array, shape=array.shape)
        return array
