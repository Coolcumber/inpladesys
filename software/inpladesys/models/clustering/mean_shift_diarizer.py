from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation
from inpladesys.datatypes import Dataset, Segmentation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from typing import List
import numpy as np
import time


class MeanShiftDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None, task=None) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        # scaling didn't help ??

        predicted_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

            x = documents_features[i]  #x_scaled[i]
            true_n_clusters = dataset.segmentations[i].author_count

            assert x.shape[0] == len(preprocessed_documents[i])

            bandwith = estimate_bandwidth(x, quantile=0.3)
            print('bandwith:', bandwith)

            diarizer = MeanShift()
            labels = diarizer.fit_predict(x)
            predicted_label_lists.append(labels)

            estimated_n_clusters = len(diarizer.cluster_centers_)

            print('Document', i + 1, '/', len(documents_features), x.shape, 'in', time.time() - start_time, 's',)
            print('Real author count = {}, estimated = {}'.format(true_n_clusters, estimated_n_clusters))
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents, task=task)

