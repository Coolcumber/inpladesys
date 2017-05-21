from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation, find_cluster_for_noisy_samples
from sklearn.preprocessing import StandardScaler
from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector
from inpladesys.models.model_selection.DBSCAN_model_selector import DBSCANModelSelector
from typing import List
import numpy as np
from inpladesys.datatypes import *
from sklearn import preprocessing
from scipy.sparse import issparse
from sklearn.cluster import DBSCAN
import time


class DBSCANDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None) -> List[Segmentation]:

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

            x_scaled = StandardScaler(with_mean=not issparse(doc_features)).fit_transform(doc_features)  #preprocessing.scale(doc_features, axis=0)
            # x_scaled = doc_features

            diarizer = DBSCAN(eps=hyperparams['eps'],
                              min_samples=hyperparams['min_samples'],
                              metric=hyperparams['metric'],
                              algorithm='brute')  # The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.

            labels = diarizer.fit_predict(x_scaled)
            estimated_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noisy = find_cluster_for_noisy_samples(labels)
            predicted_label_lists.append(labels)

            print('Document', i+1, '/', len(documents_features), x_scaled.shape, 'in', time.time()-start_time, 's',)
            print('Real author count = {}, estimated = {}, noisy = '.format(true_n_clusters,
                                                                            estimated_n_clusters), noisy)
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents)

    def get_model_selector(self) -> AbstractModelSelector:
        hyperparams = {
            'eps': np.arange(100, 1100, 100),
            'min_samples': [i for i in range(1, 100, 10)],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
        return DBSCANModelSelector(hyperparams=hyperparams, scaler=StandardScaler)




