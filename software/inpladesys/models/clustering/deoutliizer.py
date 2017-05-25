from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


class Deoutliizer():
    def __init__(self, outlier_proportion):
        self.outlier_proportion = outlier_proportion

    def fit_predict(self, X):
        X = X[:]
        m = X.shape[0]
        n = int(m * (1 - self.outlier_proportion))
        keep = np.ones(m, dtype=np.int32)
        while m > n:
            center = np.average(X, axis=0, weights=keep)
            distances = np.sum((X - center) ** 2, axis=1)
            keep[np.argmax(distances)] = 0
            m -= 1
        return keep
