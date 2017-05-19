from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from .abstract_author_diarizer import AbstractAuthorDiarizer
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.util.cacher import Cacher
from sklearn import preprocessing
import numpy as np


class PipelineAuthorDiarizer(AbstractAuthorDiarizer):
    def __init__(self, parameters: dict, cache_dir=None, random_state=-1):
        self.preprocessor = parameters['document_preprocessor']
        self.bfe = parameters['basic_feature_extractor']
        # TODO: move to bfe constructor
        # self.bfe.context_size = parameters['context_size']
        self.ft = parameters['feature_transformer']
        self.clusterer = parameters['clusterer']
        self.cacher = Cacher(cache_dir, dummy=cache_dir is None)

    def train(self, dataset: Dataset):
        docs, segmentations = dataset.documents, dataset.segmentations

        print("(1/5) Training basic feature extractor...")
        corpus = "\n\n".join(doc for doc, _ in dataset)
        preprocessed_corpus = self.preprocessor.fit_transform(corpus)
        self.bfe.fit(corpus, preprocessed_corpus)

        @self.cacher.cache()
        def preprocess_training_data():
            preprocessed_docs = []
            document_token_features = []  # [document index][token index]
            document_token_labels = []
            for i in range(dataset.size):
                doc = docs[i]
                prepr_doc = self.preprocessor.fit_transform(docs[i])
                preprocessed_docs.append(prepr_doc)
                document_token_features.append(
                    preprocessing.scale(self.bfe.transform(doc, prepr_doc)))
                o2a = segmentations[i].offsets_to_authors
                document_token_labels.append(o2a(t[1] for t in prepr_doc))
                print('Document {}/{}'.format(i + 1, dataset.size))
            return preprocessed_docs, document_token_features, document_token_labels

        print("(2/5) Preprocessing training data...")
        preprocessed_docs, document_token_features, document_token_labels = preprocess_training_data()

        print("(4/5) Training feature transformer...")
        x, y = document_token_features[:], document_token_labels[:]
        self.ft.fit(x, y)

        if True:
            import matplotlib.pyplot as plt

            x1 = x[0:1]
            y1 = y[0:1]
            for i in range(100):
                self.ft.fit(x, y)
                h = self.ft.transform(x1)[0]
                plt.clf()
                hx = h[:, 1]
                plt.scatter(h[:, 0], h[:, 1], c=y1[0])
                plt.pause(0.05)

                # h = self.ft.transform(x[0])
                # plt.scatter(h[:, 0], h[:, 1], c=y[:,0])
                # plt.show()

    def _predict(self, document: Document) -> Segmentation:
        pass  # TODO
