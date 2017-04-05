import nltk
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from datatypes import Document, Segment, Segmentation, Dataset
from .abstract_author_diarizer import AbstractAuthorDiarizer


class SimpleFixedAuthorDiarizer(AbstractAuthorDiarizer):
    def __init__(self, author_count):  # author count should be given in advance
        self.n = author_count

    def train(self, dataset: Dataset):
        pass

    def _predict(self, document: Document) -> Segmentation:
        sent_list = sent_tokenize(document)
        sent_count = len(sent_list)

        # Bag of Words feature
        vectorizer = CountVectorizer(max_features=100)
        bow = vectorizer.fit_transform(sent_list)
        x = bow

        sent_length = []  # word count per sentence
        avg_token_length = []

        univ_tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN',
                       'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']  # universal POS tagset (TODO hard coded curr.)
        pos_counts = []

        for sentence in range(sent_count):
            tokens = word_tokenize(sentence)

            # sentence length feature
            sent_length.append(len(tokens))

            # average word length feature
            len_sum = 0
            for token in tokens:
                len_sum += len(token)
            avg_token_length.append(round(len_sum / len(tokens)))

            # POS tag count feature
            pos_tags = nltk.pos_tag(tokens, tagset='universal')
            counts = Counter(tag for word, tag in pos_tags)

            count_vals = []
            for univ_tag in univ_tagset:
                count_vals.append(counts[univ_tag])

            pos_counts.append(count_vals.copy())


        # TODO merge to one vector and standardize features

        # k-means clustering
        kmeans = KMeans(n_clusters=self.n, random_state=22).fit(x)
        predicted_labels = kmeans.labels_

        segments = []
        for i in range(sent_count):
            sentence = sent_list[i]
            author = predicted_labels[i]
            length = len(sentence)
            offset = document.index(sentence)
            segments.append(Segment(offset=offset, length=length, author=author))

        return Segmentation(self.n, segments)

