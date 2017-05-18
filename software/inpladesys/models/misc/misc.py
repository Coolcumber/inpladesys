from inpladesys.datatypes import Segment, Segmentation
from typing import List
import numpy as np


def generate_segmentation(preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
             document_label_lists, documents) -> List[Segmentation]:
    assert len(documents_features) == len(preprocessed_documents)
    segmentations = []
    for i in range(len(documents_features)):
        preprocessed_doc_tokens = preprocessed_documents[i]
        doc_features = documents_features[i]
        assert doc_features.shape[0] == len(preprocessed_doc_tokens)
        labels = document_label_lists[i]
        segments = []
        for k in range(doc_features.shape[0]):
            prep_token = preprocessed_doc_tokens[k]
            segments.append(Segment(offset=prep_token[1],
                                    length=prep_token[2] - prep_token[1],
                                    author=labels[k]))
        segmentations.append(Segmentation(author_count=max(labels) + 1,
                                          segments=segments,
                                          max_repairable_error=60,
                                          document_length=len(documents[i])))
    return segmentations
