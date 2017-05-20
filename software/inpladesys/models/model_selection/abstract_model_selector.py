from abc import ABC
from abc import abstractmethod

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
