from datasets import Pan16DatasetLoader
from models import DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer, SimpleFixedAuthorDiarizer
import nltk

#nltk.download() #  uncomment if running for first time

dataset_dir = "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16"
loader = Pan16DatasetLoader(dataset_dir)
dataset = loader.load_dataset()

for model in [DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer]:
    m = model()
    m.train(dataset)
    print("model: " + str(model))
    print(m.predict(dataset.documents[0]))
    print()

s = SimpleFixedAuthorDiarizer(author_count=2)
print("model: " + str(s))
print(s.predict(dataset.documents[0]))
