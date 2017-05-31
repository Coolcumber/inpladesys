from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from inpladesys.models.clustering.auto_k_means import AutoKMeans
from inpladesys.util.cacher import Cacher
from sklearn.cluster import KMeans
from inpladesys.evaluation import get_confusion_matrix, BCubedScorer, MicroScorer, MacroScorer
import numpy as np
from inpladesys.models.misc import perform_confidence_interval_test

dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]


def evaluate(params: dict, dataset_index: int, cache_dir=None, linear=True):
    features_file_name = 'inpladesys/models/basic_feature_extraction/features_files/cng-sw-bow.json'
    random_state = 0xBeeFeed
    pl_params = dict()
    pl_params['basic_feature_extractor'] = BasicFeatureExtractor(
        features_file_name=features_file_name,
        context_size=params['context_size'])
    from inpladesys.models.feature_transformation import SimpleGroupRepelFeatureTransformer
    if not linear:
        pl_params['feature_transformer'] = SimpleGroupRepelFeatureTransformer(
            reinitialize_on_fit=False,
            iteration_count=params['gr-iteration_count'],
            learning_rate=5e-4,
            random_state=random_state)
    else:
        pl_params['feature_transformer'] = GroupRepelFeatureTransformer(
            output_dimension=params['gr_output_dimension'],
            reinitialize_on_fit=False,
            nonlinear_layer_count=params['gr-nonlinear_layer_count'],
            iteration_count=params['gr-iteration_count'],
            learning_rate=1e-3,
            random_state=random_state)
    print(str(pl_params['feature_transformer']))
    pl_params['clusterer'] = AutoKMeans(min_clusters=2, max_clusters=2) if dataset_index != 1 else AutoKMeans(2, 2)
    # pl_params['clusterer'] = Deoutliizer(0.3)
    if params['basic_feature_extender'] == 'f**2':
        pl_params['basic_feature_extender'] = lambda f: np.concatenate((f, f ** 2), axis=0)
    else:
        pl_params['basic_feature_extender'] = None

    # keys = sorted(params.keys())
    keys = ['context_size', 'basic_feature_extender']
    if cache_dir is None: cache_dir = ""
    cache_dir = ".model-cache/t-pipeline/task-{}/".format("abc"[dataset_index]) + cache_dir
    if cache_dir is None:
        cache_dir += ''.join(
            "({}={})".format(k, params[k]) for k in keys)
    pad = PipelineAuthorDiarizer(pl_params, cacher=Cacher(cache_dir))

    print("Loading dataset...")
    dataset = Pan16DatasetLoader(dataset_dirs[dataset_index]).load_dataset()
    dataset.shuffle(random_state=random_state)
    train_validate_data, _ = dataset.split(0, int(0.7 * dataset.size))
    train_data, validate_data = train_validate_data.split(0, int(0.7 * train_validate_data.size))

    print("Training...")
    pad.train(train_data)

    # validate_data = train_data
    print("Evaluating on validation data...")
    hs = pad.predict(validate_data.documents,
                     author_counts=[s.author_count for d, s in validate_data] if dataset_index == 1 else None)
    ys = validate_data.segmentations

    def get_scores(scorer_factory, y, h):
        scorer = scorer_factory(y, h)
        return np.array([scorer.precision(), scorer.recall(), scorer.f1_score()])

    scorer_factories = [MicroScorer, MacroScorer, BCubedScorer] if dataset_index == 0 else [BCubedScorer]
    for sf in scorer_factories:
        score_list = [get_scores(sf, y, h) for y, h in zip(ys, hs)]
        f1_scores = [s[2] for s in score_list]
        print(perform_confidence_interval_test(f1_scores))
        scores = np.stack(score_list, axis=0)
        avg_scores = np.average(scores, axis=0)
        score_variances = np.var(scores, axis=0)
        print(sf)
        print(avg_scores, '+-', score_variances ** 0.5)


params = dict()
params['context_size'] = 120
params['gr_output_dimension'] = 40
params['gr-nonlinear_layer_count'] = 0
params['gr-iteration_count'] = 40  # 60  #20simple
params['basic_feature_extender'] = 'f2'

# cache_dir example "ce100-bow100-sw--ctx120--f21"
evaluate(params,
         2,
         cache_dir="cng120-bow120-sw--ctx{}--f2-s".format(
            params['context_size'],
            1 if params['basic_feature_extender'] == 'f2' else 0),
         linear=True)

# Task c:
# 1) output_dimension=16, 2 groups, 100 iter, no f**2 , linear: 0.53
# 2) (1) ctx=120, f**2 : 0.58, 0.56, 0.55
# 3) (2) ctx = 160 : 0.52
# 5) (4) 1 hidden layer : 0.49
# 6) (5) 20 iter : 0.496
# 7) (2) 3 groups : 0.494
# 8) (2) +sw : 0.55
# 9) (8) output_dimension=20 : 0.556, 0.540
# 9) (8) output_dimension=24 : 0.547, 0.542
# 9) (8) output_dimension=32 : 0.560
# 9) (8) output_dimension
#
#
# =32 : 0.539
# 10) (8) output_dimension=120 : 0.547
# 11) cs120 cng120 bow120 sw atl f2 od 40 linear 0.57, weights [ 0.58638496  0.77830424  0.64190473] +- [ 0.1642388   0.14514787  0.09098742]

# 0.558 - ce 50 sw bow 100 | cs 120 od 16 f2
# 0.563 - bow 100 sw | cs 120 od 16 f2
# 0.561 - bow 200 sw | cs 120 od 16 f2
# 0.561 - bow 200 | cs 120 od 16 f2 1 iter 100
# 0.558 - bow 200 | cs 120 od 16 f2 1 iter 200
# 0.561 - bow 200 | cs 120 od 16 f2 1 iter 50

# Simple Grouprepel - sole weighting
# [ 0.59058178  0.77409589  0.64278143] +- [ 0.16430171  0.15012622  0.10052927]
# [ 0.59824322  0.81718198  0.67191475] +- [ 0.18099589  0.14361191  0.14636418]
# 1e-4 all feasible features
# ce100-bow100-sw [ 0.56613638  0.71394097  0.60896839] +- [ 0.15861691  0.12229928  0.09630066]
# ce300 bow300 [ 0.58075337  0.74768844  0.62966799] +- [ 0.16752679  0.11718516  0.09843741]
