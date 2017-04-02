from datasets import Pan16DatasetLoader
from models import DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer, SimpleFixedAuthorDiarizer
import nltk


def ellipsis(string):
    return (str(string[:500]) + '...') if len(string) > 500 else string

# nltk.download() #  uncomment if running for first time

dataset_dir = "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16"
loader = Pan16DatasetLoader(dataset_dir)
dataset = loader.load_dataset()

models = [
    (DummySingleAuthorDiarizer, "DummySingleAuthorDiarizer"),
    (DummyStochasticAuthorDiarizer, "DummyStochasticAuthorDiarizer"),
    (lambda: SimpleFixedAuthorDiarizer(author_count=2), "SimpleFixedAuthorDiarizer")
]

for model in models:
    m = model[0]()
    m.fit(dataset.documents, dataset.segmentations)
    print("model: " + model[1])
    print("output:")
    print(ellipsis(m.predict(dataset.documents[2]).to_char_sequence(0.05)))
    print("truth:")
    print(ellipsis(dataset.segmentations[2].to_char_sequence(0.05)))
    print()
