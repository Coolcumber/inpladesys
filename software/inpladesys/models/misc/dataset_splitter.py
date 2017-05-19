import numpy as np
from typing import List
from inpladesys.datatypes.dataset import Dataset
from sklearn.model_selection import train_test_split


def custom_train_test_split(preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                            dataset: Dataset, train_size, random_state):

    # indices of every document
    indices_of_docs = [i for i in range(len(preprocessed_documents))]

    i_train, i_test = train_test_split(indices_of_docs, train_size=train_size, random_state=random_state)

    prep_docs_train = [preprocessed_documents[i] for i in i_train]
    prep_docs_test = [preprocessed_documents[i] for i in i_test]

    doc_features_train = [documents_features[i] for i in i_train]
    doc_features_test = [documents_features[i] for i in i_test]

    author_counts_train = [dataset.segmentations[i].author_count for i in i_train]
    author_counts_test = [dataset.segmentations[i].author_count for i in i_test]

    documents_train = [dataset.documents[i] for i in i_train]
    documents_test = [dataset.documents[i] for i in i_test]

    return prep_docs_train, prep_docs_test, \
           doc_features_train, doc_features_test, \
           author_counts_train, author_counts_test, \
           documents_train, documents_test


