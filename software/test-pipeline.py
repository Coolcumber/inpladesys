from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from inpladesys.models.clustering.auto_k_means import AutoKMeans
from inpladesys.util.cacher import Cacher
from sklearn.cluster import KMeans
from inpladesys.evaluation import get_confusion_matrix, BCubedScorer
import numpy as np

dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]


def evaluate(params: dict, dataset_index: int):
    features_file_name = 'inpladesys/models/basic_feature_extraction/features_files/char-n-grams.json'
    random_state = 0xBeeFeed
    pl_params = dict()
    pl_params['basic_feature_extractor'] = BasicFeatureExtractor(
        features_file_name=features_file_name,
        context_size=params['context_size'])
    pl_params['feature_transformer'] = GroupRepelFeatureTransformer(
        output_dimension=params['gr_output_dimension'],
        reinitialize_on_fit=False,
        nonlinear_layer_count=params['gr-nonlinear_layer_count'],
        iteration_count=params['gr-iteration_count'],
        learning_rate=8e-4,
        random_state=random_state)  # params['model'] = KMeansDiarizer()
    pl_params['clusterer'] = AutoKMeans(min_clusters=2, max_clusters=8)

    keys = sorted(params.keys())
    cache_dir = ".model-cache/t-pipeline/" + ''.join("({}={})".format(k, params[k]) for k in keys)

    pad = PipelineAuthorDiarizer(pl_params, cacher=Cacher(cache_dir))

    print("Loading dataset...")
    dataset = Pan16DatasetLoader(dataset_dirs[dataset_index]).load_dataset()
    dataset.shuffle(random_state=random_state)
    train_validate_data, _ = dataset.split(0, int(0.7 * dataset.size))
    train_data, validate_data = train_validate_data.split(0, int(0.7 * train_validate_data.size))

    print("Training...")
    pad.train(train_data)

    print("Validation...")
    hs = pad.predict(validate_data.documents)
    ys = validate_data.segmentations

    def get_scores(y, h):
        cm = get_confusion_matrix(y, h)
        scorer = BCubedScorer(cm)
        return np.array([scorer.precision(), scorer.recall(), scorer.f1_score()])

    # print([scores(y, h) for y, h in zip(ys, hs)])
    scores = [get_scores(y, h) for y, h in zip(ys, hs)]
    print(scores)
    score = np.average(scores, axis=0)
    print(score)


# 1) output_dimension 16, 2 groups, 100 iter, no f**2 , linear: 0.53
# 2) (1) & ctx=120, f**2 : 0.58, 0.56, 0.55
# 3) (2) & ctx = 160 : 0.52
# 5) (4) & 1 hidden layer : 0.49
# 6) (5) & 20 iter : 0.496
# 7) (2) & 3 groups : 0.494

params = dict()
params['context_size'] = 120
params['gr_output_dimension'] = 16
params['gr-nonlinear_layer_count'] = 0
params['gr-iteration_count'] = 100

evaluate(params, 2)
