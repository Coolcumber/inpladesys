from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation
from sklearn.preprocessing import StandardScaler
from typing import List
import numpy as np
from inpladesys.datatypes import *
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import time


class DBSCANDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        predicted_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

            preprocessed_doc_tokens = preprocessed_documents[i]
            doc_features = documents_features[i]
            true_n_clusters = dataset.segmentations[i].author_count

            assert doc_features.shape[0] == len(preprocessed_doc_tokens)

            # svd = TruncatedSVD(n_components=50)
            # normalizer = Normalizer(copy=False)
            # lsa = make_pipeline(svd, normalizer)
            # x_scaled = lsa.fit_transform(doc_features)

            x_scaled = StandardScaler().fit_transform(doc_features)  #preprocessing.scale(doc_features, axis=0)
            # x_scaled = doc_features

            diarizer = DBSCAN(eps=6.1,
                              min_samples=5,
                              metric='manhattan',
                              algorithm='auto',  # The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
                              leaf_size=30)  # Leaf size passed to BallTree or cKDTree.

            labels = diarizer.fit_predict(x_scaled)
            predicted_label_lists.append(labels)

            estimated_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            noisy = 0
            for l_i in range(len(labels)):
                if labels[l_i] == -1:
                    labels[l_i] = 0  # TODO fix this !!!!
                    noisy += 1

            print('Document', i+1, '/', len(documents_features), 'in', time.time()-start_time, 's',)
            print('Real author count = {}, estimated = {}, noisy = '.format(true_n_clusters,
                                                                            estimated_n_clusters), noisy)
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents)
