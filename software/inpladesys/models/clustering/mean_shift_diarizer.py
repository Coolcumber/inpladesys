from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation
from inpladesys.datatypes import Dataset, Segmentation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from typing import List
import numpy as np
import time


class MeanShiftDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        x_scaled = []
        for doc_features in documents_features:
            x_scaled.append(StandardScaler().fit_transform(doc_features))

        predicted_label_lists = []

        for i in range(len(x_scaled)):
            start_time = time.time()

            x = x_scaled[i]
            true_n_clusters = dataset.segmentations[i].author_count

            assert x.shape[0] == len(preprocessed_documents[i])

            diarizer = MeanShift(bandwidth=100,
                                 bin_seeding=True)
            labels = diarizer.fit_predict(x)
            predicted_label_lists.append(labels)

            estimated_n_clusters = len(diarizer.cluster_centers_)

            print('Document', i + 1, '/', len(documents_features), x.shape, 'in', time.time() - start_time, 's',)
            print('Real author count = {}, estimated = {}'.format(true_n_clusters, estimated_n_clusters))
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents)

