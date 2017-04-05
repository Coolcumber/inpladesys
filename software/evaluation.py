from collections import namedtuple
import numpy as np
from datatypes import Segmentation, Segment


def get_confusion_matrix(seg_true: Segmentation, seg_pred: Segmentation):
    def nextsp():  # < >
        nonlocal sp, sp_end
        try:
            sp = sp_iter.__next__()
        except StopIteration:
            return -1
        sp_end = sp.offset + sp.length
        return 0
    sp_iter, sp, sp_end = iter(seg_pred.segments), None, 0
    cm = np.zeros(shape=(seg_true.author_count, seg_pred.author_count), dtype=int)
    nextsp()
    for st in seg_true.segments:
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
        self.bcms = [BinaryConfusionMatrix(self.cm, i)
                     for i in range(self.cm.shape[0])]
        self.bcm_sum = np.sum(self.bcms, axis=0)

    def wambo(self, measure):
        return measure(self.cm)

    def macro(self, measure):
        return np.average(np.array([measure(bcm) for bcm in self.bcms]), axis=0)

    def micro(self, measure):
        return measure(self.bcm_sum)

    def bCubed(self, measure):
        # http://goo.gl/HJLMNm
        # http://nlp.uned.es/docs/amigo2007a.pdf
        pass  # TODO

if False:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s = Scorer(a)
    print(s.macro(f1_score))
