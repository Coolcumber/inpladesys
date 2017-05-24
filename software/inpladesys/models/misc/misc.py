from inpladesys.datatypes import Segment, Segmentation
from typing import List
import numpy as np
from inpladesys.datatypes.dataset import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import scipy.stats as st

def generate_segmentation(preprocessed_documents: List[List[tuple]], documents_features: List[np.ndarray],
                          document_label_lists, documents, task=None) -> List[Segmentation]:
    assert len(documents_features) == len(preprocessed_documents)
    segmentations = []
    for i in range(len(documents_features)):
        preprocessed_doc_tokens = preprocessed_documents[i]
        doc_features = documents_features[i]
        assert doc_features.shape[0] == len(preprocessed_doc_tokens)
        labels = document_label_lists[i]
        segments = []
        for k in range(doc_features.shape[0]):
            prep_token = preprocessed_doc_tokens[k]
            segments.append(Segment(offset=prep_token[1],
                                    length=prep_token[2] - prep_token[1],
                                    author=labels[k]))
        segmentations.append(Segmentation(author_count=max(labels) + 1,
                                          segments=segments,
                                          max_repairable_error=60,
                                          document_length=len(documents[i])))
    if task == 'a':
        for segmentation in segmentations:
            fix_segmentation_labels_for_plagiarism_detection(segmentation)

    return segmentations


def fix_segmentation_labels_for_plagiarism_detection(segmentation, plagiarism_majority=False):
    # the majority label should be 0 (original author)
    assert segmentation.author_count == 2
    author_segments = segmentation.by_author[0]
    plagiarism_segments = segmentation.by_author[1]
    author_len = sum(s.length for s in author_segments)
    plagiarism_len = sum(s.length for s in plagiarism_segments)
    swap = author_len < plagiarism_len
    if plagiarism_majority:
        swap = not swap
    if swap:
        for s in segmentation:
            s.author = 1 - s.author
        segmentation.by_author[0] = plagiarism_segments
        segmentation.by_author[1] = author_segments


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

    dataset_train = Dataset([dataset.documents[i] for i in i_train],
                            [dataset.segmentations[i] for i in i_train])

    dataset_test = Dataset([dataset.documents[i] for i in i_test],
                           [dataset.segmentations[i] for i in i_test])

    return prep_docs_train, prep_docs_test, \
           doc_features_train, doc_features_test, \
           author_counts_train, author_counts_test, \
           dataset_train, dataset_test


def find_cluster_for_noisy_samples(predicted_labels, context_size=10):
    start = time.time()
    len_ = len(predicted_labels)
    counter = Counter(predicted_labels)
    noisy = counter[-1]
    unclustered_label = 0
    if -1 in counter.keys():
        if len(counter.most_common()) == 1:
            predicted_labels[:] = unclustered_label
        else:
            for i in range(len_):
                if predicted_labels[i] == -1:
                    left_diff = i - context_size
                    left = left_diff if left_diff >= 0 else 0
                    right_diff = i + context_size
                    right = right_diff if right_diff < len_ else len_
                    counter = Counter(predicted_labels[left:right])
                    if -1 in counter.keys():
                        if len(counter.most_common()) == 1:
                            predicted_labels[left:right] = unclustered_label
                        else:
                            found, curr = 0, 0
                            while found == 0:
                                if counter.most_common()[curr][0] != -1:
                                    predicted_labels[i] = counter.most_common()[curr][0]
                                    found = 1
                                curr += 1
    # print('Noisy labels reclustered in {}'.format(time.time()-start))
    return noisy


#  https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data/34474255#34474255
def perform_confidence_interval_test(samples: List, c_interval=0.95, p_normal_threshold=0.05):
    n = len(samples)
    if n >= 30:
        sem = st.sem(samples)
        mean = np.mean(samples)
        interval = st.t.interval(c_interval, n-1, loc=mean, scale=sem)
        print('Mean:', mean)
        print('Standard error:', sem)
        print('{}% confidence interval: {}\n'.format(c_interval*100, interval))
    else:
        #  https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.normaltest.html
        #  https://stackoverflow.com/questions/12838993/scipy-normaltest-how-is-it-used
        z, p_val = st.normaltest(samples, nan_policy='raise')
        if p_val < p_normal_threshold:
            print('A given sample is not from normal distribution: '
                  'p_val = {} < threshold = {}'.format(p_val, p_normal_threshold))
            print('The confidence intervals cannot be calculated.')
        else:
            sem = st.sem(samples)
            mean = np.mean(samples)
            interval = st.t.interval(c_interval, n - 1, loc=mean, scale=sem)
            print('Mean:', mean)
            print('Standard error:', sem)
            print('{}% confidence interval: {}\n'.format(c_interval * 100, interval))
