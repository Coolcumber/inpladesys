import json
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models import SimpleFixedAuthorDiarizer
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor


class LearningPipeline:

    def __init__(self, parameters: dict):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.basic_feature_extractor = parameters['basic_feature_extractor']  # TODO rename to basic
        self.feature_postprocessor = parameters['feature_postprocessor']
        self.model = parameters['model']

    def do_chain(self):
        for i in range(1): #self.dataset.size
            document, segmentation = self.dataset[i]
            preprocessed_document = self.document_preprocessor.fit_transform(document)
            self.basic_feature_extractor.fit(document, preprocessed_document)
            document_features = self.basic_feature_extractor.transform(document, preprocessed_document,
                                                                       self.context_size)
            print(document_features)
            # TODO split dataset into train, validation and test set if needed here ??

            # TODO model, evaluation, log



if True:
    dataset_dirs = [
        "../../../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
    ]

    print("Loading dataset...")
    dataset = Pan16DatasetLoader(dataset_dirs[1]).load_dataset()

    features_file_name = 'basic_feature_extraction/features_files/features1m.json'

    params = dict()
    params['dataset'] = dataset
    params['context_size'] = 20
    params['document_preprocessor'] = TokenizerPreprocessor()
    params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
    params['feature_postprocessor'] = None
    params['model'] = SimpleFixedAuthorDiarizer(author_count=3)  # TODO remove author_count from constructor

    pipeline = LearningPipeline(params)
    pipeline.do_chain()






