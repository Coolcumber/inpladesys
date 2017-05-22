from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.evaluation import get_confusion_matrix, BCubedScorer, MacroScorer, MicroScorer
from inpladesys.models import (DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer,
                               SimpleFixedAuthorDiarizer)
from inpladesys.models.misc import fix_segmentation_labels_for_plagiarism_detection
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
dataset_index = 0
dataset = Pan16DatasetLoader(dataset_dirs[dataset_index]).load_dataset()

models = [
    (DummySingleAuthorDiarizer, "DummySingleAuthorDiarizer"),
    (lambda: DummyStochasticAuthorDiarizer(author_count=2), "DummyStochasticAuthorDiarizer"),
    (lambda: SimpleFixedAuthorDiarizer(author_count=2), "SimpleFixedAuthorDiarizer")
]


def name_model(factory, name):
    def nc(): return factory()

    nc.name = name
    nc.__str__ = lambda self: self.name
    return nc


models = [name_model(*m) for m in models]

if False:
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
        scorer = BCubedScorer(cm)
        print("BCubed precision:", scorer.precision())
        print("BCubed recall:", scorer.recall())
        print("BCubed F1 score:", scorer.f1_score())

print("Testing on the dataset...")
modelsToScores = dict(
    (m.name, np.array([0, 0, 0], dtype=float)) for m in models)
for model in models:
    m = model()
    m.fit(dataset.documents, dataset.segmentations)
    for d, truth in zip(dataset.documents, dataset.segmentations):
        if isinstance(m, SimpleFixedAuthorDiarizer):
            if dataset_index != 2:  # tasks a and b
                m.author_count = truth.author_count
            else:  # task c
                m.choose_author_count(d)
        pred = m.predict(d)
        if dataset_index == 0:  # task a
            #micro = MicroScorer(truth, pred)  # use Micro AND Macro Scorer for task a!!
            macro = MacroScorer(truth, pred)  # use Micro AND Macro Scorer for task a!!
            modelsToScores[model.name] += np.array([macro.precision(), macro.recall(), macro.f1_score()])

        else:  # tasks b and c
            scorer = BCubedScorer(truth, pred)
            modelsToScores[model.name] += np.array([scorer.precision(),
                                                    scorer.recall(), scorer.f1_score()])

for model in models:
    modelsToScores[model.name] /= len(dataset.documents)
    print(model.name)
    print(modelsToScores[model.name])
