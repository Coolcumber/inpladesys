import numpy as np

from datatypes import Segmentation


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
    cm = np.zeros(shape=(seg_true.author_count, seg_pred.author_count), dtype=int)
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

### TODO

class BinaryConfusionMatrix(np.ndarray): # TODO
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


def accuracy(cm): return np.trace(cm) / sum(cm)


def precision(bcm): return bcm.tp() / (bcm.tp() + bcm.fp() + 2e-16)


def recall(bcm): return bcm.tp() / (bcm.tp() + bcm.fn() + 2e-16)


def f1_score(bcm):
    p, r = precision(bcm), recall(bcm)
    return 2 * p * r / (p + r + 2e-16)


def fa_score(bcm, a):
    p, r = precision(bcm), recall(bcm)
    a *= a
    return 2 * (1 + a) * p * r / (a * p + r + 2e-16)


class Scorer(): # TODO
    def __init__(self, confusion_matrix):
        self.cm = confusion_matrix
        self.extended_cm = confusion_matrix

        extra_cols = self.cm.shape[0] - self.cm.shape[1]
        extra_rows = -extra_cols

        if extra_cols > 0:
            extra = np.zeros((self.cm.shape[0], extra_cols))
            self.extended_cm = np.hstack((self.cm, extra))

        elif extra_rows > 0:
            extra = np.zeros((extra_rows, self.cm.shape[1]))
            self.extended_cm = np.vstack((self.cm, extra))

        self.bcms = [BinaryConfusionMatrix(self.extended_cm, i)
                     for i in range(self.extended_cm.shape[0])]
        self.bcm_sum = np.sum(self.bcms, axis=0)

    def wambo(self, measure):
        return measure(self.cm)

    def macro(self, measure):
        return np.average(np.array([measure(bcm) for bcm in self.bcms]), axis=0)

    def micro(self, measure):
        return measure(self.bcm_sum)

    def bCubed(self, measure: str):
        def precision(cm):
            cm_sum = np.sum(cm)
            squared_cm = cm ** 2
            numerators = np.sum(squared_cm, axis=0)
            denominators = np.sum(cm, axis=0)
            return np.sum(numerators / denominators) / cm_sum

        def recall(cm):
            cm_sum = np.sum(cm)
            squared_cm = cm ** 2
            numerators = np.sum(squared_cm, axis=1)
            denominators = np.sum(cm, axis=1)
            return np.sum(numerators / denominators) / cm_sum

        def f1_score(cm):
            p, r = precision(cm), recall(cm)
            return 2 * p * r / (p + r + 2e-16)

        if measure == 'precision':
            return precision(self.cm)

        elif measure == 'recall':
            return recall(self.cm)

        elif measure == 'f1_score':
            return f1_score(self.cm)

        else:
            message = 'Unknown measure: {}'.format(measure)
            raise ValueError(message)

if True:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s = Scorer(a)
    print(s.macro(f1_score))

    cm = np.array([[4, 1, 0],
                   [0, 2, 4],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

    s = Scorer(cm)
    print(s.bCubed('precision'))
    print(s.bCubed('recall'))
    print(s.bCubed('f1_score'))






