from numpy.ctypeslib import prep_array

from inpladesys.datatypes import Document, Segment, Segmentation, Dataset
from .abstract_author_diarizer import AbstractAuthorDiarizer
from inpladesys.models.misc import generate_segmentation, fix_segmentation_labels_for_plagiarism_detection
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.util.cacher import Cacher
from sklearn import preprocessing
import numpy as np


class PipelineAuthorDiarizer():
    def __init__(self, parameters: dict, cacher=None):
        self.preprocessor = TokenizerPreprocessor()
        self.bfextr = parameters['basic_feature_extractor']
        self.bf_extender = parameters['basic_feature_extender']
        self.feature_transformer = parameters['feature_transformer']
        self.clusterer = parameters['clusterer']
        self.cacher = Cacher.dummy() if cacher is None else cacher
        self.bfe_trained = False

    def train(self, dataset: Dataset):
        @self.cacher("preprocessed-training-labels")
        def get_bydoc_labels(bydoc_tokens, segmentations):
            o2a = lambda i, offset: segmentations[i].offsets_to_authors(offset)
            return [o2a(i, (t[1] for t in tokens)) for i, tokens in enumerate(bydoc_tokens)]

        @self.cacher("bfe-trained")
        def fit_bfe(documents):
            corpus = "\n\n".join(doc for doc in documents)
            corpus_tokens = self.preprocessor.fit_transform(corpus)
            self.bfextr.fit(corpus, corpus_tokens)
            return self.bfextr

        documents, segmentations = dataset.documents, dataset.segmentations

        print("(1/4) Training basic feature extractor...")
        self.bfextr = fit_bfe(documents)  # TODO: load bfe in predict if not loaded

        print("(2/4) Preprocessing training data: extracting tokens and basic features...")
        bydoc_tokens, bydoc_features = self._preprocess_documents(documents)

        print("(3/4) Preprocessing training data: assigning labels to tokens...")
        bydoc_labels = get_bydoc_labels(bydoc_tokens, segmentations)

        print("(4/4) Training feature transformer...")
        #bydoc_features = [preprocessing.scale(f) for f in bydoc_features]
        X, Y = bydoc_features[:], bydoc_labels[:]
        if True:
            self.feature_transformer.fit(X, Y)
        else:
            import matplotlib.pyplot as plt
            self.feature_transformer.iteration_count //= 100
            self.feature_transformer.iteration_count += 1
            x1 = X[0:1]
            y1 = Y[0:1]
            for i in range(100):
                self.feature_transformer.fit(X, Y)
                h = self.feature_transformer.transform(x1)[0]
                plt.clf()
                hx = h[:, 1]
                plt.scatter(h[:, 0], h[:, 1], c=y1[0])
                plt.pause(0.05)

        print("(5/4) Training clusterer...")
        if getattr(self.clusterer, "train", None) is not None:
            self.clusterer.train(self.feature_transformer.transform(X), [s.author_count for s in segmentations])

    def predict(self, documents, author_counts=None):
        assert (len(documents) > 0)
        bydoc_tokens, bydoc_features = self._preprocess_documents(documents)
        bydoc_transformed_features = self.feature_transformer.transform(bydoc_features)
        bydoc_labels_h = []
        if author_counts is None:
            bydoc_labels_h = [self.clusterer.fit_predict(tf) for tf in bydoc_transformed_features]
        else:
            for i, tf in enumerate(bydoc_transformed_features):
                bydoc_labels_h.append(self.clusterer.fit_predict(tf, cluster_count=author_counts[i]))

        segms = generate_segmentation(bydoc_tokens, bydoc_features, bydoc_labels_h, documents)
        if all(segm.author_count <= 2 for segm in segms):
            for segm in segms:
                fix_segmentation_labels_for_plagiarism_detection(segm)
        return segms

    def _preprocess_documents(self, documents):
        bydoc_tokens = []
        bydoc_token_features = []  # [document index][token index]
        for i in range(len(documents)):
            doc = documents[i]
            import hashlib
            doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()

            @self.cacher("preprocessed-document-{}-{}".format(i, doc_hash))
            def prepr():
                return self._preprocess_document(doc)

            tokens, token_features = prepr()
            token_features = np.array(token_features)
            bydoc_tokens.append(tokens)
            bydoc_token_features.append(token_features)
            print("\r{}/{}".format(i + 1, len(documents)), end='')
        print('')
        return bydoc_tokens, bydoc_token_features

    def _preprocess_document(self, document):
        tokens = self.preprocessor.fit_transform(document)
        features = self.bfextr.transform(document, tokens)
        features = [np.concatenate((f, f ** 2), axis=0) for f in features]  # move scaling to corpus level
        return tokens, features  # scaling beneficial
