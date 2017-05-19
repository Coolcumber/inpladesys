from inpladesys.models.abstract_diarizer import AbstractDiarizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from inpladesys.datatypes import *
from typing import List
import numpy as np


class KMeansDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]],
                    documents_features: List[np.ndarray],
                    dataset: Dataset) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        segmentations = []

        for i in range(len(documents_features)):
            preprocessed_doc_tokens = preprocessed_documents[i]
            doc_features = documents_features[i]
            num_authors = dataset.segmentations[i].author_count

            assert doc_features.shape[0] == len(preprocessed_doc_tokens)

            #svd = TruncatedSVD(n_components=50)
            #normalizer = Normalizer(copy=False)
            #lsa = make_pipeline(svd, normalizer)
            #x_scaled = lsa.fit_transform(doc_features)

            x_scaled = preprocessing.scale(doc_features, axis=0)
            #x_scaled = doc_features

            diarizer = KMeans(n_clusters=num_authors,
                            init='k-means++',
                            n_init=10,
                            max_iter=300,
                            algorithm='auto',
                            verbose=0)  # “auto” chooses “elkan” for dense data and “full” (EM style) for sparse data.

            labels = diarizer.fit_predict(x_scaled)

            segments = []

            for k in range(doc_features.shape[0]):
                author = labels[k]
                prep_token = preprocessed_doc_tokens[k]
                offset = prep_token[1]
                length = prep_token[2] - prep_token[1]
                segments.append(Segment(offset=offset, length=length, author=author))

            segmentations.append(Segmentation(num_authors, segments, max_repairable_error=60, document_length=len(dataset.documents[i])))

        return segmentations


