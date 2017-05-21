from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from inpladesys.util.cacher import Cacher
from sklearn.cluster import KMeans
from inpladesys.evaluation import get_confusion_matrix, BCubedScorer
import numpy as np

dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]

print("Loading dataset...")
dataset = Pan16DatasetLoader(dataset_dirs[2]).load_dataset()

features_file_name = 'inpladesys/models/basic_feature_extraction/features_files/features1m.json'

params = dict()
params['basic_feature_extractor'] = BasicFeatureExtractor(
    features_file_name, context_size=30)
params['feature_transformer'] = GroupRepelFeatureTransformer(
    output_dimension=2,
    reinitialize_on_fit=False,
    nonlinear_layer_count=0,
    iteration_count=10,
    learning_rate=1e-3)  # params['model'] = KMeansDiarizer()
params['clusterer'] = KMeans(n_clusters=5)

pad = PipelineAuthorDiarizer(params, cacher=Cacher(".pipeline-cache"))
dataset.shuffle(order_determining_number=1337)
train_data, test_data = dataset.split(0, int(0.7 * dataset.size))
train_data, validation_data = train_data.split(0, int(0.7 * train_data.size))
pad.train(train_data)
hs = pad.predict(validation_data.documents)
ys = validation_data.documents

def scores(y, h):
    cm = get_confusion_matrix(y, h)
    scorer = BCubedScorer(cm)
    f1 = scorer.f1_score()
    return np.array([scorer.precision(), scorer.recall(), scorer.f1_score()])

#print([scores(y, h) for y, h in zip(ys, hs)])

score = np.average([scores(y, h) for y, h in zip(ys, hs)], axis=0)
print(score)
