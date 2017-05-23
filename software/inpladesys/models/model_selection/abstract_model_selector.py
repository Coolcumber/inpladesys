from abc import ABC
from abc import abstractmethod
from sklearn import metrics
from inpladesys.evaluation import *


class AbstractModelSelector(ABC):

    def __init__(self, hyperparams, model=None, scorer=None, scaler=None):
        self.model = model
        self.hyperparams = hyperparams
        self.scorer = scorer
        self.scaler = scaler

    @abstractmethod
    def select_optimal_hyperparams(self, preprocessed_documents, documents_features, documents,
                                   true_segmentations, author_labels=None, author_counts=None):
        return None


    def get_bcubed_f1(self, true_segmentations, predicted_segmentations):
        assert len(true_segmentations) == len(predicted_segmentations)
        result = 0

        for i in range(len(true_segmentations)):
            truth = true_segmentations[i]
            pred = predicted_segmentations[i]
            result += BCubedScorer(truth, pred).f1_score()

        return result / len(true_segmentations)

    def get_silhouette_coeff(self, x_scaled, predicted_labels_lists, metric):
        score = 0
        for i in range(len(x_scaled)):
            if len(set(predicted_labels_lists[i])) == 1:
                return -2
            score += metrics.silhouette_score(x_scaled[i], predicted_labels_lists[i], metric)
        return score / len(x_scaled)

    def get_calinski_harabaz_score(self, x_scaled, predicted_labels_lists):
        score = 0
        for i in range(len(x_scaled)):
            if len(set(predicted_labels_lists[i])) == 1:
                return -2
            score += metrics.calinski_harabaz_score(x_scaled[i], predicted_labels_lists[i])
        return score / len(x_scaled)

    def get_esstimated_n_difference(self, predicted_labels_lists, author_counts):
        diff = 0.1
        for i in range(len(predicted_labels_lists)):
            labels = predicted_labels_lists[i]
            estimated_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            diff += abs(estimated_n_clusters - author_counts[i])
        return diff

    def get_macro_f1(self, true_segmentations, predicted_segmentations):
        assert len(true_segmentations) == len(predicted_segmentations)
        result = 0

        for i in range(len(true_segmentations)):
            truth = true_segmentations[i]
            pred = predicted_segmentations[i]
            result += MacroScorer(truth, pred).f1_score()

        return result / len(true_segmentations)
