from inpladesys.models.abstract_diarizer import AbstractDiarizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from inpladesys.datatypes import *
from typing import List
import numpy as np
import time
from inpladesys.models.misc.misc import generate_segmentation


class KMeansDiarizer(AbstractDiarizer):

    def fit_predict(self, preprocessed_documents: List[List[tuple]],
                    documents_features: List[np.ndarray],
                    dataset: Dataset, hyperparams=None, task=None) -> List[Segmentation]:

        assert len(documents_features) == len(preprocessed_documents)

        document_label_lists = []

        for i in range(len(documents_features)):
            start_time = time.time()

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
            document_label_lists.append(labels)

            print('Document', i + 1, '/', len(documents_features), 'in', time.time() - start_time, 's')

        return generate_segmentation(preprocessed_documents, documents_features,
                                     document_label_lists, dataset.documents, task=task)


