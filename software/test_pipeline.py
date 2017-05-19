from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.feature_transformation import GroupRepelFeatureTransformer
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
from  inpladesys.util.cacher import Cacher
from sklearn.cluster import KMeans

dataset_dirs = [
    "../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
    "../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
]

print("Loading dataset...")
dataset = Pan16DatasetLoader(dataset_dirs[2]).load_dataset()

features_file_name = 'inpladesys/models/basic_feature_extraction/features_files/features1m.json'

params = dict()

params['document_preprocessor'] = TokenizerPreprocessor()
params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name, context_size=30)
params['feature_transformer'] = GroupRepelFeatureTransformer(
    output_dimension=2,
    reinitialize_on_fit=False,
    nonlinear_layer_count=0,
    iteration_count=10,
    learning_rate=2e-3)  # params['model'] = KMeansDiarizer()
params['clusterer'] = KMeans()

pad = PipelineAuthorDiarizer(params, cache_dir=".pipeline-cache")
dataset.shuffle(order_determining_number=1337)
train_data, test_data = dataset.split(0, 2+0*int(0.7 * dataset.size))
# train_data, validation_data = train_data.split(0, int(0.7*train_data.size))
pad.train(train_data)
