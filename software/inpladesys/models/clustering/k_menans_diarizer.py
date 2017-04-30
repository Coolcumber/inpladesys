from inpladesys.models.abstract_author_diarizer import AbstractAuthorDiarizer
from sklearn.cluster import KMeans
from sklearn import preprocessing
from inpladesys.datatypes import *
from typing import List
import numpy as np


class KMeansDiarizer(AbstractAuthorDiarizer):

    def fit_predict(self, dataset: Dataset, documents_features: List[np.ndarray],
                    preprocessed_documents: List[tuple] = None) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        segmentations = []

        for i in range(len(documents_features)):
            preprocessed_doc_tokens = preprocessed_documents[i]
            doc_features = documents_features[i]
            num_authors = dataset.segmentations[i].author_count

            assert doc_features.shape[0] == len(preprocessed_doc_tokens)

            x_scaled = preprocessing.scale(doc_features, axis=0)

            kmeans = KMeans(n_clusters=num_authors,
                            init='k-means++',
                            n_init=10,
                            max_iter=300,
                            algorithm='auto',
                            verbose=0)  # “auto” chooses “elkan” for dense data and “full” (EM style) for sparse data.

            labels = kmeans.fit_predict(x_scaled)

            segments = []

            for k in range(doc_features.shape[0]):
                author = labels[k]
                prep_token = preprocessed_doc_tokens[k]
                offset = prep_token[1]
                length = prep_token[2] - prep_token[1]
                segments.append(Segment(offset=offset, length=length, author=author))

            segmentations.append(Segmentation(num_authors, segments, maxRepairableError=60, document_length=len(dataset.documents[i])))

        return segmentations





    def _predict(self, document: Document, features: np.ndarray = None) -> Segmentation:
        return super()._predict(document, features)


