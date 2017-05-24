from inpladesys.models.abstract_diarizer import AbstractDiarizer
from inpladesys.models.misc.misc import generate_segmentation
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from typing import List
import numpy as np
from inpladesys.datatypes import *
import time


class IsolationForestDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None, task=None) -> List[Segmentation]:

        x_scaled = []

        svd = TruncatedSVD(n_components=2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        for doc_features in documents_features:
            #x_scaled.append(StandardScaler().fit_transform(doc_features))
            x_scaled.append(lsa.fit_transform(doc_features))

        predicted_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

            x = x_scaled[i]  # documents_features[i]  x_scaled[i]

            assert x.shape[0] == len(preprocessed_documents[i])

            diarizer = IsolationForest(n_estimators=100, max_samples=1.0,
                                       contamination=0.3, max_features=1.0,
                                       bootstrap=True, random_state=None)

            diarizer.fit(x)
            labels_array = diarizer.predict(x)
            labels_array[labels_array == 1] = 0
            labels_array[labels_array == -1] = 1
            predicted_label_lists.append(labels_array.tolist())

            print('Document', i + 1, '/', len(documents_features), x.shape, 'in', time.time() - start_time, 's', )
            print()

        return generate_segmentation(preprocessed_documents, documents_features,
                                     predicted_label_lists, dataset.documents, task=task)

