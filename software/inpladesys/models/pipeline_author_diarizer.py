from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple
from .abstract_author_diarizer import AbstractAuthorDiarizer
import numpy as np
import random
import time
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer


class PipelineAuthorDiarizer(AbstractAuthorDiarizer):

    def __init__(self, parameters: dict, random_state=-1):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.bfe = parameters['basic_feature_extractor']
        self.f_transformer = parameters['feature_transformer']
        self.model = parameters['model']

    def train(self, dataset: Dataset):
        print("Training basic feature extractor...")
        corpus = "\n\n\n".join(doc for doc, _ in self.dataset)
        preprocessed_corpus = self.document_preprocessor.fit_transform(
            corpus)
        self.bfe.fit(corpus, preprocessed_corpus)

        print("Preparing training data for the feature transformer...")
        docs_features = []
        preprocessed_docs = []
        for i in range(self.dataset.size):  # self.dataset.size
            document, _ = self.dataset[i]
            preprocessed_doc = self.document_preprocessor.fit_transform(
                document)
            document_features = self.bfe.transform(
                document, preprocessed_doc, self.context_size)
            print('Document {}/{}: {}'.format(i + 1,
                                              self.dataset.size, document_features.shape))
            docs_features.append(document_features)
            preprocessed_docs.append(preprocessed_doc)
        tokenwise_labelses = [[None]] #TODO

        print("Training feature transformer...")
        bfe_feat_count = self.bfe.transform(
            self.dataset[0][0], preprocessed_docs[0], self.context_size).shape[0]
        self.f_transformer = GroupRepelFeatureTransformer(
            input_dimension=bfe_feat_count,
            output_dimension=8,
            nonlinear_layer_count=3,
            iteration_count=1000)
        transform_train_set= Dataset(preprocessed_docs[:], tokenwise_labelses[:])
        for i in range(1000):
            transform_train_set.shuffle()
            self.f_transformer.fit(*transform_train_set)

    def _predict(self, document: Document) -> Segmentation:
        pass  # TODO
