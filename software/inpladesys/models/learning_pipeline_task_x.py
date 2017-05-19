import time
import numpy as np
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.evaluation import *
from inpladesys.evaluation import get_confusion_matrix
from inpladesys.models.clustering.k_means_diarizer import KMeansDiarizer
from inpladesys.models.clustering.hac_diarizer import AgglomerativeDiarizer
from inpladesys.models.clustering.dbscan_diarizer import DBSCANDiarizer


class LearningPipeline:

    def __init__(self, parameters: dict):
        self.dataset = parameters['dataset']
        self.context_size = parameters['context_size']
        self.document_preprocessor = parameters['document_preprocessor']
        self.basic_feature_extractor = parameters['basic_feature_extractor']
        self.feature_transformer = parameters['feature_transformer']
        self.model = parameters['model']
        self.scorer_1 = parameters['scorer_class_1']
        self.scorer_2 = parameters['scorer_class_2']

    def do_chain(self):
        documents_features = []

        print('Extracting features...')
        start_time = time.time()

        preprocessed_docs = []
        dataset_size = 1 #self.dataset.size

        for i in range(dataset_size):
            document, segmentation = self.dataset[i]
            preprocessed_document = self.document_preprocessor.fit_transform(document)
            self.basic_feature_extractor.fit(document, preprocessed_document)
            document_features = self.basic_feature_extractor.transform(document, preprocessed_document,
                                                                       self.context_size)
            print('Document {}/{}: {}'.format(i+1, dataset_size, document_features.shape))

            # TODO postprocess features (transformations etc.)

            documents_features.append(document_features)
            preprocessed_docs.append(preprocessed_document)

        print('Extraction time (s): ', time.time() - start_time)

        if True:

            print('Running model..')
            # TODO is it better to use fit and _predict separately ??
            pred_segmentations = self.model.fit_predict(preprocessed_docs, documents_features, self.dataset)

            # TODO postprocess model results

            # Evaluation
            print('Evaluating...')
            results_1 = np.array([0, 0, 0], dtype=np.float64)
            results_2 = np.array([0, 0, 0], dtype=np.float64)

            for i in range(dataset_size):
                truth = self.dataset.segmentations[i]
                pred = pred_segmentations[i]

                scorer_1 = self.scorer_1(get_confusion_matrix(truth, pred))
                results_1 += np.array([scorer_1.recall(), scorer_1.precision(), scorer_1.f1_score()])

                if self.scorer_2 is not None:
                    scorer_2 = self.scorer_2(get_confusion_matrix(truth, pred))
                    results_2 += np.array([scorer_2.recall(), scorer_2.precision(), scorer_2.f1_score()])

            results_1 /= dataset_size
            print(self.scorer_1, ':', results_1)

            if self.scorer_2 is not None:
                results_2 /= dataset_size
                print(self.scorer_2, ':', results_2)

        # TODO write all necessary params to log file



if True:
    dataset_dirs = [
        "../../../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
    ]

    features_file_name = 'basic_feature_extraction/features_files/features1m.json'
    params = dict()

    # Change the task here
    task = 'c'

    if task == 'a':
        print("Loading dataset for task ", task, "...")
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[0]).load_dataset()
        params['context_size'] = 16
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = AgglomerativeDiarizer()  # AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = MicroScorer
        params['scorer_class_2'] = MacroScorer

    elif task == 'b':
        print("Loading dataset for task ", task, "...")
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[1]).load_dataset()
        params['context_size'] = 140  # 140 for kmeans
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = AgglomerativeDiarizer() #AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = BCubedScorer
        params['scorer_class_2'] = None

    elif task == 'c':
        print("Loading dataset for task ", task, "...")
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[2]).load_dataset()
        params['context_size'] = 140
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = DBSCANDiarizer()  # AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = BCubedScorer
        params['scorer_class_2'] = None

    pipeline = LearningPipeline(params)
    pipeline.do_chain()






