from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple
from .abstract_author_diarizer import AbstractAuthorDiarizer
import numpy as np
import random
import time
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer


class PipelineAuthorDiarizer(AbstractAuthorDiarizer):
    def __init__(self, parameters: dict, random_state=-1):
        self.preprocessor = parameters['document_preprocessor']
        self.bfe = parameters['basic_feature_extractor']
        self.bfe_context_size = parameters['context_size']
        self.ft = parameters['feature_transformer']
        self.segmentation_generator = parameters['model']

    def train(self, dataset: Dataset):
        print("(1/5) Training basic feature extractor...")
        corpus = "\n\n".join(doc for doc, _ in dataset)
        preprocessed_corpus = self.preprocessor.fit_transform(corpus)
        self.bfe.fit(corpus, preprocessed_corpus)

        print("(2/5) Preprocessing training data...")
        document_token_features = []
        preprocessed_docs = []
        for i in range(dataset.size):
            doc, _ = dataset[i]
            preprocessed_doc = self.preprocessor.fit_transform(doc)
            preprocessed_docs.append(preprocessed_doc)
            doc_features = self.bfe.transform(
                doc, preprocessed_doc, self.bfe_context_size)
            print('Document {}/{}'.format(i + 1, dataset.size))
            document_token_features.append(doc_features)
        document_token_labels = [[None]]  # TODO:

        print("(4/5) Training feature transformer...")
        basic_feature_count = self.bfe.transform(
            dataset[0][0], preprocessed_docs[0], self.bfe_context_size).shape[0]
        self.ft = GroupRepelFeatureTransformer(
            input_dimension=basic_feature_count,
            output_dimension=8,
            nonlinear_layer_count=3,
            iteration_count=1000)
        transform_train_set = Dataset(
            document_token_features[:], document_token_labels[:])
        self.ft.fit(*transform_train_set)

    def _predict(self, document: Document) -> Segmentation:
        pass  # TODO
