import numpy as np
import abc
from inpladesys.datatypes import Segmentation
from abc import ABC

Epsilon = 2e-16


def get_confusion_matrix(seg_true: Segmentation, seg_pred: Segmentation):
    def nextsp():  # < >
        nonlocal sp, sp_end
        try:
            sp = sp_iter.__next__()
        except StopIteration:
            return -1
        sp_end = sp.offset + sp.length
        return 0

    sp_iter, sp, sp_end = iter(seg_pred), None, 0
    cm = np.zeros(shape=(seg_true.author_count, seg_pred.author_count),
                  dtype=int)
    nextsp()
    for st in seg_true:
        st_end = st.offset + st.length
        while True:
            if sp.offset < st.offset:
                if sp_end < st_end:  # .<[>].
                    cm[st.author, sp.author] += sp.length - \
                                                (st.offset - sp.offset)
                    if nextsp() == -1:
                        return cm
                else:  # .<[]>.
                    cm[st.author, sp.author] += st.length
                    break
            else:
                if sp_end < st_end:  # [<>]
                    cm[st.author, sp.author] += sp.length
                    if nextsp() == -1:
                        return cm
                else:  # # [<]>
                    cm[st.author, sp.author] += sp.length - (sp_end - st_end)
                    break
    return cm


class BinaryConfusionMatrix(np.ndarray):  # TODO
    def __new__(cls, conf_mat, i):
        obj = np.zeros((2, 2)).view(cls)
        obj[0, 0] = conf_mat[i, i]  # TP
        obj[0, 1] = np.sum(conf_mat[i, :]) - obj[0, 0]  # FN
        obj[1, 0] = np.sum(conf_mat[:, i]) - obj[0, 0]  # FP
        obj[1, 1] = np.sum(conf_mat) - obj[0, 0] - obj[0, 1] - obj[1, 0]  # TN
        return obj

    def tp(self): return self[0, 0]

    def fn(self): return self[0, 1]

    def fp(self): return self[1, 0]

    def tn(self): return self[1, 1]


class BinaryScorer:
    @staticmethod
    def accuracy(cm): return np.trace(cm) / sum(cm)

    @staticmethod
    def precision(bcm): return bcm.tp() / (bcm.tp() + bcm.fp() + Epsilon)

    @staticmethod
    def recall(bcm): return bcm.tp() / (bcm.tp() + bcm.fn() + Epsilon)

    @staticmethod
    def f1_score(bcm):
        p, r = BinaryScorer.precision(bcm), BinaryScorer.recall(bcm)
        return 2 * p * r / (p + r + Epsilon)

    @staticmethod
    def fa_score(bcm, a):
        p, r = BinaryScorer.precision(bcm), BinaryScorer.recall(bcm)
        a *= a
        return 2 * (1 + a) * p * r / (a * p + r + Epsilon)


class AbstractScorer(ABC):
    def __init__(self, confusion_matrix):
        self.cm = confusion_matrix

    @abc.abstractmethod
    def precision(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def recall(self):
        raise NotImplementedError()

    def f1_score(self):
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r + Epsilon)

    def fa_score(self, a):
        p, r = self.precision(), self.recall()
        a *= a
        return 2 * (1 + a) * p * r / (a * p + r + Epsilon)


class MicroScorer(AbstractScorer):
    def __init__(self, square_confusion_matrix):
        super().__init__(square_confusion_matrix)
        self.bcms = [BinaryConfusionMatrix(self.cm, i)
                     for i in range(self.cm.shape[0])]
        self.bcm_sum = BinaryConfusionMatrix(np.sum(self.bcms, axis=0), 0)

    def recall(self):
        return BinaryScorer.recall(self.bcm_sum)

    def precision(self):
        return BinaryScorer.precision(self.bcm_sum)


class MacroScorer(AbstractScorer):
    def __init__(self, square_confusion_matrix):
        super().__init__(square_confusion_matrix)
        self.bcms = [BinaryConfusionMatrix(self.cm, i)
                     for i in range(self.cm.shape[0])]

    def recall(self):
        return sum(BinaryScorer.recall(bcm) for bcm in self.bcms) / len(
            self.bcms)

    def precision(self):
        return sum(BinaryScorer.precision(bcm) for bcm in self.bcms) / len(
            self.bcms)

    def f1_score(self):
        return sum(BinaryScorer.f1_score(bcm) for bcm in self.bcms) / len(
            self.bcms)

    def fa_score(self, a):
        return sum(BinaryScorer.fa_score(bcm, a) for bcm in self.bcms) / len(
            self.bcms)


class BCubedScorer(AbstractScorer):
    def __init__(self, confusion_matrix):
        super().__init__(confusion_matrix)
        self.cm_sum = np.sum(self.cm)
        self.squared_cm = self.cm ** 2

    def recall(self):
        numerators = np.sum(self.squared_cm, axis=1)
        denominators = np.sum(self.cm, axis=1)
        return np.sum(numerators / denominators) / self.cm_sum

    def precision(self):
        numerators = np.sum(self.squared_cm, axis=0)
        denominators = np.sum(self.cm, axis=0)
        return np.sum(numerators / (denominators + Epsilon)) / self.cm_sum


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(MacroScorer(a).f1_score())
    print()

    cm = np.array([[4, 1, 0],
                   [0, 2, 4],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

    bc = BCubedScorer(cm)
    print(bc.precision())
    print(bc.recall())
    print(bc.f1_score())
    print()

    s = np.array([[1, 3], [2, 4]])

    m = MicroScorer(s)
    print(m.precision())
    print(m.recall())
    print(m.f1_score())
    print()

    M = MacroScorer(s)
    print(M.precision())
    print(M.recall())
    print(M.f1_score())
