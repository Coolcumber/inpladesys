import time
import numpy as np
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models import SimpleFixedAuthorDiarizer
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.evaluation import BCubedScorer
from inpladesys.evaluation import get_confusion_matrix


class LearningPipeline:

    def __init__(self, parameters: dict):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.basic_feature_extractor = parameters['basic_feature_extractor']  # TODO rename to basic
        self.feature_postprocessor = parameters['feature_postprocessor']
        self.model = parameters['model']

    def do_chain(self):
        documents_features = []

        start_time = time.time()

        for i in range(self.dataset.size): #self.dataset.size
            document, segmentation = self.dataset[i]
            preprocessed_document = self.document_preprocessor.fit_transform(document)
            self.basic_feature_extractor.fit(document, preprocessed_document)
            document_features = self.basic_feature_extractor.transform(document, preprocessed_document,
                                                                       self.context_size)
            print(document_features.shape)

            # TODO postprocess features (transformations etc.)

            documents_features.append(document_features)

        print('Extraction time (s): ', time.time() - start_time)

        # TODO is it better to use fit and _predict separately ??
        pred_segmentations = self.model.fit_predict(self.dataset, documents_features)

        # TODO postprocess model results

        # TODO evaluate results
        results = np.array([0, 0, 0])
        for i in range(self.dataset.size):
            truth = self.dataset.segmentations[i]
            pred = pred_segmentations[i]

            # use Micro and Macro scorer for task a, and BCubed for tasks b and c
            bc = BCubedScorer(get_confusion_matrix(truth, pred))
            results += np.array([bc.precision(), bc.recall(), bc.f1_score()])

        results /= self.dataset.size
        print(results)

        # TODO write all necessary params to log file



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
    params['context_size'] = 100
    params['document_preprocessor'] = TokenizerPreprocessor()
    params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
    params['feature_postprocessor'] = None
    params['model'] = SimpleFixedAuthorDiarizer(author_count=3)  # TODO remove author_count from constructor

    pipeline = LearningPipeline(params)
    pipeline.do_chain()






