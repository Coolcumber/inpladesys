from datasets import Pan16DatasetLoader
from evaluation import *
from evaluation import get_confusion_matrix, Scorer
from models import (DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer,
                    SimpleFixedAuthorDiarizer)


def ellipsis(string):
    return (str(string[:500]) + '...') if len(string) > 500 else string

# nltk.download() #  uncomment if running for first time


dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]
print("Loading dataset...")
dataset = Pan16DatasetLoader(dataset_dirs[2]).load_dataset()

models = [
    (DummySingleAuthorDiarizer, "DummySingleAuthorDiarizer"),
    (DummyStochasticAuthorDiarizer, "DummyStochasticAuthorDiarizer"),
    (lambda: SimpleFixedAuthorDiarizer(author_count=7), "SimpleFixedAuthorDiarizer")
]

print("Running models...")
docind = 5
for model in models:
    m = model[0]()
    m.fit(dataset.documents, dataset.segmentations)
    pred = m.predict(dataset.documents[docind])
    truth = dataset.segmentations[docind]
    print("model: " + model[1])
    print("output:")
    print(ellipsis(pred.to_char_sequence(0.05)))
    print("truth:")
    print(ellipsis(truth.to_char_sequence(0.05)))
    print("confusion matrix:")
    cm = get_confusion_matrix(truth, pred)
    print(cm)
    s = Scorer(cm)
    print("BCubed precision:", s.bCubed("precision"))
    print("BCubed recall:", s.bCubed("recall"))
    print("BCubed F1 score:", s.bCubed("f1_score"))

