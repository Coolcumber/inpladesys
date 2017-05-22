from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from .abstract_author_diarizer import AbstractAuthorDiarizer
from inpladesys.models.misc import generate_segmentation
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.util.cacher import Cacher
from sklearn import preprocessing
import numpy as np


class PipelineAuthorDiarizer():
    def __init__(self, parameters: dict, cacher=None):
        self.preprocessor = TokenizerPreprocessor()
        self.bfe = parameters['basic_feature_extractor']
        self.feature_transformer = parameters['feature_transformer']
        self.clusterer = parameters['clusterer']
        self.cacher = Cacher.dummy() if cacher is None else cacher
        self.bfe_trained = False

    def train(self, dataset: Dataset):
        @self.cacher("preprocessed-training-datapoints")
        def _preprocess_documents(documents):
            return self._preprocess_documents(documents)

        @self.cacher("preprocessed-training-labels")
        def get_bydoc_labels(bydoc_tokens, segmentations):
            o2a = lambda i, offset: segmentations[i].offsets_to_authors(offset)
            return [o2a(i, (t[1] for t in tokens)) for i, tokens in enumerate(bydoc_tokens)]

        @self.cacher("bfe-trained")
        def fit_bfe(documents):
            corpus = "\n\n".join(doc for doc in documents)
            corpus_tokens = self.preprocessor.fit_transform(corpus)
            self.bfe.fit(corpus, corpus_tokens)
            return self.bfe

        documents, segmentations = dataset.documents, dataset.segmentations

        print("(1/4) Training basic feature extractor...")
        self.bfe = fit_bfe(documents)  # TODO: load bfe in predict if not loaded

        print("(2/4) Preprocessing training data: extracting tokens and basic features...")
        bydoc_tokens, bydoc_features = _preprocess_documents(documents)

        print("(3/4) Preprocessing training data: assigning labels to tokens...")
        bydoc_labels = get_bydoc_labels(bydoc_tokens, segmentations)

        print("(4/4) Training feature transformer...")
        x, y = bydoc_features[:], bydoc_labels[:]
        if True:
            self.feature_transformer.fit(x, y)
        else:
            import matplotlib.pyplot as plt
            self.feature_transformer.iteration_count //= 100
            self.feature_transformer.iteration_count += 1
            x1 = x[0:1]
            y1 = y[0:1]
            for i in range(100):
                self.feature_transformer.fit(x, y)
                h = self.feature_transformer.transform(x1)[0]
                plt.clf()
                hx = h[:, 1]
                plt.scatter(h[:, 0], h[:, 1], c=y1[0])
                plt.pause(0.05)

    def predict(self, documents):
        assert (len(documents) > 0)
        bydoc_tokens, bydoc_features = self._preprocess_documents(documents)
        bydoc_transformed_features = self.feature_transformer.transform(bydoc_features)
        bydoc_labels_h = [self.clusterer.fit_predict(tf) for tf in bydoc_transformed_features]
        return generate_segmentation(bydoc_tokens, bydoc_features, bydoc_labels_h, documents)

    def _preprocess_documents(self, documents):
        bydoc_tokens = []
        bydoc_token_features = []  # [document index][token index]
        for i in range(len(documents)):
            doc = documents[i]
            tokens, tokens_features = self._preprocess_document(doc)
            bydoc_tokens.append(tokens)
            bydoc_token_features.append(tokens_features)
            print("\r{}/{}".format(i + 1, len(documents)), end='')
        print('')
        return bydoc_tokens, bydoc_token_features

    def _preprocess_document(self, document):
        tokens = self.preprocessor.fit_transform(document)
        features = self.bfe.transform(document, tokens)
        features = [np.concatenate((f, f ** 2), axis=0) for f in features]
        return tokens, preprocessing.scale(features)


"""class PipelineAuthorDiarizerFactory():
    def __init__(self, parameters: dict, cacher=None, random_state=-1):
        from sklearn.cluster import KMeans
        from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
        self.preprocessor = parameters['document_preprocessor']
        self.bfe = parameters['basic_feature_extractor']
        self.feature_transformer = GroupRepelFeatureTransformer()
        self.clusterer = KMeans()
        self.cacher = Cacher.dummy()
        self.bfe_trained = False"""
