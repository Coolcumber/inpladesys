from abc import ABC, abstractmethod
from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from typing import List, Tuple


class AbstractAuthorDiarizer(ABC):  # TODO
    # scikit.learn-compatible interface - either override this or train
    def fit(self, documents: List[Document], segmentations: List[Segmentation]):
        """
        Fit (or train) is used to learn a feature transformation for better 
        clustering.
        """
        self.train(Dataset(documents, segmentations))
    
    # alternative interface - either override this or fit
    def train(self, dataset: Dataset):
        self.fit(dataset.documents, dataset.segmentations)
    
    @abstractmethod
    def _predict(self, document: Document) -> Segmentation:
        pass

    # Do not override if not really necessary
    def predict(self, doc) -> List[Segmentation]:
        if type(doc) is Document:
            return self._predict(doc)
        return [self._predict(d) for d in doc]


