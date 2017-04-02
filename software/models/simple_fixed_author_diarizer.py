import nltk

from nltk import sent_tokenize
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

        # Bag of Words feature
        vectorizer = CountVectorizer(max_features=100)
        bow = vectorizer.fit_transform(sent_list)
        x = bow

        #  TODO add more features and standardize them

        # k-means clustering
        kmeans = KMeans(n_clusters=self.n, random_state=22).fit(x)
        predicted_labels = kmeans.labels_

        segments = []
        for i in range(len(sent_list)):
            sentence = sent_list[i]
            author = predicted_labels[i]
            length = len(sentence)
            offset = document.index(sentence)
            segments.append(Segment(offset=offset, length=length, author=author))

        return Segmentation(self.n, segments)

