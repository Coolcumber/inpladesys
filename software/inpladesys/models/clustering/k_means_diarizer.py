from inpladesys.models.abstract_diarizer import AbstractDiarizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from inpladesys.datatypes import *
from typing import List
import numpy as np
import time
from inpladesys.models.misc.misc import generate_segmentation
from matplotlib import pyplot as plt


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

            # remove
            #truth = dataset.segmentations[i].offsets_to_authors([j for j in range(len(preprocessed_doc_tokens))])
            #print(truth)

            assert doc_features.shape[0] == len(preprocessed_doc_tokens)


            #svd = TruncatedSVD(n_components=2)
            #normalizer = Normalizer(copy=False)
            #lsa = make_pipeline(svd, normalizer)
            #x_scaled = lsa.fit_transform(doc_features)

            #plt.scatter(x_scaled[:,0], x_scaled[:,1], s=100,  c=truth)
            #plt.show()

            x_scaled = StandardScaler().fit_transform(doc_features) #preprocessing.scale(doc_features, axis=0)  # StandardScaler().fit_transform(doc_features)
            #x_scaled = doc_features

            use_one_cluster = (task == 'a')

            diarizer = KMeans(n_clusters=num_authors, #if use_one_cluster else num_authors,
                              init='k-means++',
                              n_init=10,
                              max_iter=300,
                              algorithm='full')  # “auto” chooses “elkan” for dense data and “full” (EM style) for sparse data.

            labels = diarizer.fit_predict(x_scaled)

            #if use_one_cluster:
            #    diffs = []
            #    threshold = 1
            #    inertia = diarizer.inertia_
            #    avg_inertia = inertia / doc_features.shape[0]
            #    centroid = diarizer.cluster_centers_[0].reshape((1, x_scaled.shape[1]))
            #    for k in range(len(labels)):
            #        x_d = x_scaled[k].reshape((1, x_scaled[k].shape[0]))
            #        inertia_x = paired_distances(x_d, centroid, metric='euclidean') ** 2 #np.sum(sq_diff)
            #        diffs.append(inertia_x)
            #        if inertia_x[0] > threshold * avg_inertia:
            #            labels[k] = 1
            #    print('min:', min(diffs))
            #    print('max:', max(diffs))
            #    print('avg:', avg_inertia)
            #    print()

            document_label_lists.append(labels)

            print('Document', i + 1, '/', len(documents_features), 'in', time.time() - start_time, 's')

        return generate_segmentation(preprocessed_documents, documents_features,
                                     document_label_lists, dataset.documents, task=task)


