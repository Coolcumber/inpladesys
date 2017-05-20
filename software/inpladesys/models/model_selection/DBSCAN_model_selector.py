from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector
from inpladesys.models.misc.misc import generate_segmentation, find_cluster_for_noisy_samples
from sklearn.cluster import DBSCAN
from inpladesys.evaluation import *
from scipy.sparse import issparse
import time


class DBSCANModelSelector(AbstractModelSelector):

    def select_optimal_hyperparams(self, preprocessed_documents, documents_features, documents,
                                   true_segmentations, author_labels=None, author_counts=None):
        x_scaled = []
        for doc_features in documents_features:
            x_scaled.append(self.scaler(with_mean=not issparse(doc_features)).fit_transform(doc_features))

        #true_labels = train_set['']

        results = []
        total_hyperparams = len(self.hyperparams['metric']) * len(self.hyperparams['eps']) * len(self.hyperparams['min_samples'])
        current_comb = 1
        start_time = time.time()
        for metric in self.hyperparams['metric']:
            for eps in self.hyperparams['eps']:
                for min_samples in self.hyperparams['min_samples']:

                    print('Combination {}/{}'.format(current_comb, total_hyperparams))
                    predicted_label_lists = []
                    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

                    for i in range(len(x_scaled)):
                        x = x_scaled[i]
                        labels = model.fit_predict(x)

                        find_cluster_for_noisy_samples(labels)

                        predicted_label_lists.append(labels)

                    current_comb += 1

                    predicted_segmentations = generate_segmentation(preprocessed_documents, documents_features,
                                                         predicted_label_lists, documents)

                    score = self.get_bcubed_f1(true_segmentations, predicted_segmentations)

                    results.append({'eps': eps, 'min_samples': min_samples,
                                    'metric': metric, 'score': score})

        sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
        best_result = sorted_results[0]
        print('The best hyperparams found:', best_result, 'in {} s.'.format(time.time()-start_time))
        return best_result

    def get_bcubed_f1(self, true_segmentations, predicted_segmentations):
        assert len(true_segmentations) == len(predicted_segmentations)
        result = 0

        for i in range(len(true_segmentations)):
            truth = true_segmentations[i]
            pred = predicted_segmentations[i]
            result += BCubedScorer(get_confusion_matrix(truth, pred)).f1_score()

        return result / len(true_segmentations)










