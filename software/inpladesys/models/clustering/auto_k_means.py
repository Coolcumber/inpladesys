from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


class AutoKMeans():
    def __init__(self, min_clusters=1, max_clusters=8, tol_sel=1e-3, tol=1e-4,
                 n_init_sel=5, n_init=10, max_iter_sel=100, max_iter=300,
                 verbose=False):
        self.min_clusters, self.max_clusters = min_clusters, max_clusters
        self.tol_sel, self.tol = tol_sel, tol
        self.n_init_sel, self.n_init = n_init_sel, n_init
        self.max_iter_sel, self.max_iter = max_iter_sel, max_iter
        self.verbose = verbose
        self.k_ = 1

    def fit_predict(self, X, cluster_count=None):
        if cluster_count is not None:
            km = KMeans(n_clusters=cluster_count).fit(X)
            return km.labels_
        ks = np.arange(self.min_clusters, self.max_clusters + 1)
        js = np.zeros(len(ks))
        labs = []
        for i in range(len(ks)):
            km = KMeans(n_clusters=ks[i]).fit(X)
            js[i] = km.inertia_
            labs += [km.labels_]
        jsks = js * ks
        #print(jsks)
        #plt.plot(ks, js)
        self.k_ = np.argmin(jsks)
        if self.verbose: print("The best k is " + str(self.k_) + ".")
        return labs[self.k_]
