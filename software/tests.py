from datasets import Pan16DatasetLoader

dataset_dir = "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16"
loader = Pan16DatasetLoader(dataset_dir)
dataset = loader.load_dataset()

from models import DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer

for model in [DummySingleAuthorDiarizer, DummyStochasticAuthorDiarizer]:
    m = model()
    m.train(dataset)
    print("model: " + str(model))
    print(m.predict(dataset.documents[0]))
    print()