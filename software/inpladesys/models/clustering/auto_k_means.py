from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from inpladesys.evaluation import BCubedScorer


class AutoKMeans():
    def __init__(self, min_clusters=1, max_clusters=8, p=1, tol_sel=1e-3, tol=1e-4,
                 n_init_sel=5, n_init=10, max_iter_sel=100, max_iter=300,
                 verbose=False):
        self.min_clusters, self.max_clusters = min_clusters, max_clusters
        self.tol_sel, self.tol = tol_sel, tol
        self.n_init_sel, self.n_init = n_init_sel, n_init
        self.max_iter_sel, self.max_iter = max_iter_sel, max_iter
        self.verbose = verbose
        self.k_ = 1
        self.p = p

    def train(self, X, ks):
        old_max_iter = self.max_iter
        self.max_iter = 200
        ps = np.linspace(-1.5, 0.5, 21)
        errors = []
        l1errors = []
        hitses = []
        for p in ps:
            self.p = p
            error = 0
            l1error = 0
            hits = 0
            for x, k in zip(X, ks):
                self.fit_predict(x)
                error += (k - self.k_) ** 2
                l1error += abs(k - self.k_)
                hits += 1 if k == self.k_ else 0
            errors.append(error)
            l1errors.append(error)
            hitses.append(hits / len(ks))
            print(errors) # 43
            print(l1errors)
            print(hitses)
        print(ps)
        self.p = ps[np.argmin(errors)]  # 0.3
        print("AutoKMeans-p {}".format(self.p))
        self.max_iter = old_max_iter

    def fit_predict(self, X, cluster_count=None):
        #cluster_count = 2
        if cluster_count is not None:
            km = KMeans(n_clusters=cluster_count).fit(X)
            return km.labels_
        ks = np.arange(self.min_clusters - 1, self.max_clusters + 2)
        js = np.zeros(len(ks))
        labs = []
        for i in range(len(ks)):
            km = KMeans(n_clusters=ks[i], max_iter=self.max_iter).fit(X)
            js[i] = km.inertia_
            labs += [km.labels_]
        dehs = np.array(
            [0.] + [-(js[i] - js[i - 1]) / (js[i + 1] - js[i]) * (i + ks[0]) ** self.p for i in
                    range(1, len(ks) - 1)] + [0.])

        # jsks = js * ks
        # print(jsks)
        #plt.plot(ks, dehs)
        i = np.argmin(dehs)
        self.k_ = ks[i]
        if self.verbose: print("The best k is " + str(self.k_) + ".")
        return labs[i]
