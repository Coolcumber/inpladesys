from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.models.pipeline_author_diarizer import PipelineAuthorDiarizer
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
# params['dataset'] = dataset
# params['context_size'] = 100
params['document_preprocessor'] = TokenizerPreprocessor()
params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name, context_size=50)
params['feature_transformer'] = None
# params['model'] = KMeansDiarizer()
params['model'] = None
pad = PipelineAuthorDiarizer(params)
dataset.shuffle(order_determining_number=1337)
dataset = dataset[0:10]
pad.train(dataset)
