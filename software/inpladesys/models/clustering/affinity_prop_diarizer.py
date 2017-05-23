from inpladesys.models.model_selection.affinity_prop_model_selector import AffinityPropModelSelector
from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector
from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation
from inpladesys.datatypes import Dataset, Segmentation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from typing import List
import numpy as np
import time


class AffinityPropDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None, task=None) -> List[Segmentation]:
        assert len(documents_features) == len(preprocessed_documents)

        x_scaled = []
        for doc_features in documents_features:
            x_scaled.append(StandardScaler().fit_transform(doc_features))

        predicted_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

            x = documents_features[i]  # documents_features[i]  x_scaled[i]
            true_n_clusters = dataset.segmentations[i].author_count

            assert x.shape[0] == len(preprocessed_documents[i])

            diarizer = AffinityPropagation(damping=hyperparams['damping'],
                                           preference=hyperparams['preference'],
                                           copy=True, affinity='euclidean',
                                           max_iter=100, convergence_iter=5)

            labels = diarizer.fit_predict(x).tolist()
            predicted_label_lists.append(labels)

            estimated_n_clusters = len(set(labels))

            print('Document', i + 1, '/', len(documents_features), x.shape, 'in', time.time() - start_time, 's', )
            print('Real author count = {}, estimated = {}'.format(true_n_clusters, estimated_n_clusters))
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents, task=task)

    def get_model_selector(self) -> AbstractModelSelector:
        hyperparams = {
            'damping': [0.9, 0.95, 0.97],
            'preference': [-28750, -28800, -29000, -30000]
        }
        return AffinityPropModelSelector(hyperparams=hyperparams, scaler=StandardScaler)

    def get_optimal_hyperparams(self):
        return {'damping': 0.97, 'preference': 28800}



