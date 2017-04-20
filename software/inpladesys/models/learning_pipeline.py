import json
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models import SimpleFixedAuthorDiarizer
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor


class LearningPipeline:

    def __init__(self, parameters: dict):
        self.dataset = parameters['dataset']
        self.document_preprocessor = parameters['document_preprocessor']
        self.feature_extractor = parameters['feature_extractor']  # TODO rename to basic
        self.feature_postprocessor = parameters['feature_postprocessor']
        self.model = parameters['model']

    def do_chain(self):  # TODO split dataset into train, validation and test set if needed
        for i in range(self.dataset.size):
            document, segmentation = self.dataset[i]
            preprocessed_doc = self.document_preprocessor.fit_transform(document)
            #self.feature_extractor.extract_features(preprocessed_doc)



if True:
    dataset_dirs = [
        "../../../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
    ]
    print("Loading dataset...")

    dataset = Pan16DatasetLoader(dataset_dirs[1]).load_dataset()

    features_file_name = 'basic_feature_extraction/features_files/features1m.json'
    j = json.load(open(features_file_name, 'r'))

    params = dict()
    params['dataset'] = dataset
    params['document_preprocessor'] = TokenizerPreprocessor()
    params['feature_extractor'] = None
    params['feature_postprocessor'] = None
    params['model'] = SimpleFixedAuthorDiarizer(author_count=3)  # TODO remove author_count from constructor


    pipeline = LearningPipeline(params)
    pipeline.do_chain()






