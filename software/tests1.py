from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.evaluation import get_confusion_matrix, BCubedScorer
from inpladesys.models import (DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer,
                    SimpleFixedAuthorDiarizer)
import numpy as np


def ellipsis(string):
    return (str(string[:500]) + '...') if len(string) > 500 else string

# nltk.download() #  uncomment if running for first time


dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]
print("Loading dataset...")
dataset = Pan16DatasetLoader(dataset_dirs[1]).load_dataset()

models = [
    (DummySingleAuthorDiarizer, "DummySingleAuthorDiarizer"),
    (DummyStochasticAuthorDiarizer, "DummyStochasticAuthorDiarizer"),
    (lambda: SimpleFixedAuthorDiarizer(author_count=4), "SimpleFixedAuthorDiarizer")
]


def named(factory, name):
    def nc(): return factory()
    nc.name = name
    nc.__str__ = lambda self: self.name
    return nc


models = [
    (DummySingleAuthorDiarizer, "DummySingleAuthorDiarizer"),
    (DummyStochasticAuthorDiarizer, "DummyStochasticAuthorDiarizer"),
    (lambda: SimpleFixedAuthorDiarizer(author_count=4), "SimpleFixedAuthorDiarizer")
]
models = [named(*m) for m in models]

print("Testing on a single document...")
docind = 8
for model in models:
    m = model()
    m.fit(dataset.documents, dataset.segmentations)
    pred = m.predict(dataset.documents[docind])
    truth = dataset.segmentations[docind]
    print("model: " + model.name)
    print("output:")
    print(ellipsis(pred.to_char_sequence(0.05)))
    print("truth:")
    print(ellipsis(truth.to_char_sequence(0.05)))
    print("confusion matrix:")
    cm = get_confusion_matrix(truth, pred)
    print(cm)
    bc = BCubedScorer(cm)
    print("BCubed precision:", bc.precision())
    print("BCubed recall:", bc.recall())
    print("BCubed F1 score:", bc.f1_score())

print("Testing on the dataset...")
modelsToScores = dict(
    (m.name, np.array([0, 0, 0], dtype=float)) for m in models)
for model in models:
    m = model()
    m.fit(dataset.documents, dataset.segmentations)
    for d in dataset.documents:
        pred = m.predict(dataset.documents[docind])
        truth = dataset.segmentations[docind]
        bc = BCubedScorer(get_confusion_matrix(truth, pred))
        modelsToScores[model.name] += np.array([bc.precision(),
                                                bc.recall(), bc.f1_score()])

for model in models:
    modelsToScores[model.name] /= len(dataset.documents)
    print(model.name)
    print(modelsToScores[model.name])
