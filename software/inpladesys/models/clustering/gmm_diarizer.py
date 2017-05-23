from inpladesys.models.abstract_diarizer import AbstractDiarizer
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from inpladesys.datatypes import *
from typing import List
import numpy as np
import time
from inpladesys.models.misc.misc import generate_segmentation
from sklearn.preprocessing import StandardScaler


class GaussianMixtureDiarizer(AbstractDiarizer):
    def fit_predict(self, preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None) -> List[Segmentation]:
        assert len(documents_features) == len(preprocessed_documents)

        document_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

            preprocessed_doc_tokens = preprocessed_documents[i]
            doc_features = documents_features[i]
            num_authors = dataset.segmentations[i].author_count

            assert doc_features.shape[0] == len(preprocessed_doc_tokens)

            # svd = TruncatedSVD(n_components=50)
            # normalizer = Normalizer(copy=False)
            # lsa = make_pipeline(svd, normalizer)
            # x_scaled = lsa.fit_transform(doc_features)

            x_scaled = StandardScaler().fit_transform(doc_features)  # preprocessing.scale(doc_features, axis=0)
            # x_scaled = doc_features

            diarizer = GaussianMixture(n_components=num_authors, covariance_type='full',
                                       tol=0.001, reg_covar=1e-06, max_iter=100,
                                       n_init=10, init_params='kmeans', weights_init=None,
                                       means_init=None, precisions_init=None, random_state=None,
                                       warm_start=False, verbose=0, verbose_interval=10)

            diarizer.fit(x_scaled)
            labels = diarizer.predict(x_scaled)
            document_label_lists.append(labels)

            print('Document', i + 1, '/', len(documents_features), 'in', time.time() - start_time, 's')

        return generate_segmentation(preprocessed_documents, documents_features,
                                     document_label_lists, dataset.documents)