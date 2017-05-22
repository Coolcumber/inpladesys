from inpladesys.models.model_selection.abstract_model_selector import AbstractModelSelector
from inpladesys.models.misc.misc import generate_segmentation
from sklearn.cluster import AgglomerativeClustering
import time


class AgglomerativeModelSelector(AbstractModelSelector):

    def select_optimal_hyperparams(self, preprocessed_documents, documents_features,
                                   documents, true_segmentations, author_labels=None,
                                   author_counts=None):
        x_scaled = []
        for doc_features in documents_features:
            x_scaled.append(self.scaler().fit_transform(doc_features))

        results = []
        total_hyperparams = len(self.hyperparams['affinity']) * len(self.hyperparams['linkage'])

        current_comb = 1
        start_time = time.time()

        for affinity in self.hyperparams['affinity']:
            for linkage in self.hyperparams['linkage']:

                    print('Combination {}/{}'.format(current_comb, total_hyperparams))

                    predicted_label_lists = []

                    for i in range(len(documents_features)):
                        model = AgglomerativeClustering(n_clusters=author_counts[i],
                                                        affinity=affinity,
                                                        linkage=linkage)

                        x = x_scaled[i]  # documents_features[i]  x_scaled[i]
                        labels = model.fit_predict(x)
                        predicted_label_lists.append(labels)

                    current_comb += 1

                    predicted_segmentations = generate_segmentation(preprocessed_documents, documents_features,
                                                                    predicted_label_lists, documents)

                    score = self.get_macro_f1(true_segmentations, predicted_segmentations)

                    results.append({'affinity': affinity, 'linkage': linkage, 'score': score})

        sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
        best_result = sorted_results[0]
        print('The best hyperparams found:', best_result, 'in {} s.'.format(time.time() - start_time))
        return best_result
