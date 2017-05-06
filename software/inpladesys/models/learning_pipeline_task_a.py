import time
import numpy as np
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.evaluation import *
from inpladesys.evaluation import get_confusion_matrix
from inpladesys.models.clustering.k_menans_diarizer import KMeansDiarizer


class LearningPipeline:

    def __init__(self, parameters: dict):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.basic_feature_extractor = parameters['basic_feature_extractor']  # TODO rename to basic
        self.feature_transformer = parameters['feature_transformer']
        self.model = parameters['model']

    def do_chain(self):
        documents_features = []

        print('Extracting features...')
        start_time = time.time()

        preprocessed_docs = []

        for i in range(self.dataset.size):  #self.dataset.size
            document, segmentation = self.dataset[i]
            preprocessed_document = self.document_preprocessor.fit_transform(document)
            self.basic_feature_extractor.fit(document, preprocessed_document)
            document_features = self.basic_feature_extractor.transform(document, preprocessed_document,
                                                                       self.context_size)
            print('Document {}/{}: {}'.format(i+1, self.dataset.size, document_features.shape))

            # TODO postprocess features (transformations etc.)

            documents_features.append(document_features)
            preprocessed_docs.append(preprocessed_document)

        print('Extraction time (s): ', time.time() - start_time)

        if True:

            print('Running model..')
            # TODO is it better to use fit and _predict separately ??
            pred_segmentations = self.model.fit_predict(self.dataset, documents_features, preprocessed_docs)

            # TODO postprocess model results


            # TODO evaluate results
            print('Evaluating...')
            results = np.array([0, 0, 0], dtype=np.float64)
            for i in range(self.dataset.size):  #self.dataset.size
                truth = self.dataset.segmentations[i]
                pred = pred_segmentations[i]

                # use Micro and Macro scorer for task a, and BCubed for tasks b and c
                bc = MacroScorer(get_confusion_matrix(truth, pred))
                results += np.array([bc.recall(), bc.precision(), bc.f1_score()])

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
    dataset = Pan16DatasetLoader(dataset_dirs[0]).load_dataset()

    features_file_name = 'basic_feature_extraction/features_files/features1m.json'

    params = dict()
    params['dataset'] = dataset
    params['context_size'] = 14
    params['document_preprocessor'] = TokenizerPreprocessor()
    params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
    params['feature_transformer'] = None
    params['model'] = KMeansDiarizer()

    pipeline = LearningPipeline(params)
    pipeline.do_chain()






