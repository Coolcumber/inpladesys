from data import Pan16DatasetLoader
from util import directory, file

dataset_dir = "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16"
loader = Pan16DatasetLoader(dataset_dir)
foo = loader.load_segmentations()
for s in foo[10]:
    print(s)