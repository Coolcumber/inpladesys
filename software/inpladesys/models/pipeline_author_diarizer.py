from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple
from .abstract_author_diarizer import AbstractAuthorDiarizer
import numpy as np
import random


class PipelineAuthorDiarizer(AbstractAuthorDiarizer):
    def __init__(self, parameters: dict, random_state=-1):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.basic_feature_extractor = parameters['basic_feature_extractor']  # TODO rename to basic
        self.feature_transformer = parameters['feature_transformer']
        self.model = parameters['model']

    def train(self, dataset: Dataset):
        pass  # TODO

    def _predict(self, document: Document) -> Segmentation:
        pass  # TODO

