from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector
from inpladesys.models.misc.misc import generate_segmentation
from sklearn.cluster import DBSCAN
from inpladesys.evaluation import *

import time

class DBSCANModelSelector(AbstractModelSelector):

    def select_optimal_hyperparams(self, preprocessed_documents, documents_features, documents,
                                   true_segmentations, author_labels=None, author_counts=None):
        x_scaled = []

        for doc_features in documents_features:
            x_scaled.append(self.scaler.fit_transform(doc_features))

        #true_labels = train_set['']

        results = []

        start_time = time.time()
        for metric in self.hyperparams['metric']:
            for eps in self.hyperparams['eps']:
                for min_samples in self.hyperparams['min_samples']:

                    predicted_label_lists = []
                    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

                    for i in range(len(x_scaled)):
                        x = x_scaled[i]
                        labels = model.fit_predict(x)

                        # TODO what with this ?
                        non_clustered_label = max(labels) + 1
                        for l in range(len(labels)):
                            if labels[l] == -1:
                                labels[l] = non_clustered_label

                        predicted_label_lists.append(labels)

                    predicted_segmentations = generate_segmentation(preprocessed_documents, documents_features,
                                                         predicted_label_lists, documents)

                    score = self.get_bcubed_f1(true_segmentations, predicted_segmentations)

                    results.append({'eps': eps, 'min_samples': min_samples,
                                    'metric': metric, 'score': score})

        sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
        print('The best hyperparams found:', sorted_results[0], 'in {} s.'.format(time.time()-start_time))

    def get_bcubed_f1(self, true_segmentations, predicted_segmentations):
        assert len(true_segmentations) == len(predicted_segmentations)
        result = 0

        for i in len(true_segmentations):
            truth = true_segmentations[i]
            pred = predicted_segmentations[i]
            result += BCubedScorer(get_confusion_matrix(truth, pred)).f1_score()

        return result / len(true_segmentations)










