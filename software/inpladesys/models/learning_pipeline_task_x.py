import time
import numpy as np
from inpladesys.datasets import Pan16DatasetLoader
from inpladesys.models.preprocessors.basic_preprocessors import TokenizerPreprocessor, BasicTokenizerPreprocessor
from inpladesys.models.basic_feature_extraction.basic_feature_extractor import BasicFeatureExtractor
from inpladesys.evaluation import *
from inpladesys.models.misc import fix_segmentation_labels_for_plagiarism_detection
from inpladesys.models.clustering.k_means_diarizer import KMeansDiarizer
from inpladesys.models.clustering.hac_diarizer import AgglomerativeDiarizer
from inpladesys.models.clustering.dbscan_diarizer import DBSCANDiarizer
from inpladesys.models.clustering.gmm_diarizer import GaussianMixtureDiarizer
from inpladesys.models.clustering.mean_shift_diarizer import MeanShiftDiarizer
from inpladesys.models.clustering.affinity_prop_diarizer import AffinityPropDiarizer
from inpladesys.models.outlier_detection.isolation_forest_diarizer import IsolationForestDiarizer
from inpladesys.models.misc.misc import custom_train_test_split
from inpladesys.util.cacher import Cacher
from sklearn import preprocessing as prep


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
        self.cacher = parameters['cacher']
        self.select_model = params['select-model']
        self.train_size = params['train_size']
        self.random_state = params['random_state']
        self.task = params['task']

        self.dataset_size = self.dataset.size

    def extract_features(self):
        @self.cacher('doc-features-prep-docs')
        def extract(self):
            print('Extracting features...')
            start_time = time.time()

            documents_features = []
            preprocessed_docs = []

            for i in range(self.dataset_size):
                document, segmentation = self.dataset[i]
                preprocessed_document = self.document_preprocessor.fit_transform(document)
                self.basic_feature_extractor.fit(document, preprocessed_document)
                document_features = self.basic_feature_extractor.transform(document, preprocessed_document,
                                                                           self.context_size)
                print('Document {}/{}: {}'.format(i + 1, self.dataset_size, document_features.shape))

                # TODO postprocess features (transformations etc.)

                documents_features.append(document_features)
                preprocessed_docs.append(preprocessed_document)

            print('Extraction time (s): ', time.time() - start_time)
            return documents_features, preprocessed_docs
        return extract(self)

    def do_chain(self):
        documents_features, preprocessed_docs = self.extract_features()

        prep_docs_train, prep_docs_test, \
            doc_features_train, doc_features_test, \
            author_counts_train, author_counts_test, \
            dataset_train, dataset_test = custom_train_test_split(preprocessed_docs, documents_features, self.dataset,
                                                                  train_size=self.train_size, random_state=self.random_state)

        print('Train set size: {}'.format(len(doc_features_train)))

        if self.select_model:
            print('Selecting model...')
            model_selector = self.model.get_model_selector()
            optimal_hyperparams = model_selector.select_optimal_hyperparams(prep_docs_train,
                                                                            doc_features_train,
                                                                            dataset_train.documents,
                                                                            dataset_train.segmentations,
                                                                            author_counts=author_counts_train,
                                                                            task=self.task)
        else:
            optimal_hyperparams = self.model.get_optimal_hyperparams(task=self.task)

        print('Running model..')
        pred_segmentations = self.model.fit_predict(prep_docs_test, doc_features_test,
                                                    dataset_test, optimal_hyperparams, task=self.task)

        # TODO postprocess model results

        # Evaluation
        print('Evaluating...')
        results_1 = np.array([0, 0, 0], dtype=np.float64)
        results_2 = np.array([0, 0, 0], dtype=np.float64)
        test_set_size = len(dataset_test)
        assert test_set_size == len(pred_segmentations)

        for i in range(test_set_size):
            truth = dataset_test.segmentations[i]
            pred = pred_segmentations[i]

            scorer_1 = self.scorer_1(truth, pred)
            results_1 += np.array([scorer_1.precision(), scorer_1.recall(), scorer_1.f1_score()])

            if self.scorer_2 is not None:  # Task a
                scorer_2 = self.scorer_2(truth, pred)
                results_2 += np.array([scorer_2.precision(), scorer_2.recall(), scorer_2.f1_score()])

        results_1 /= test_set_size
        print(self.scorer_1, ':', results_1)

        if self.scorer_2 is not None:
            results_2 /= test_set_size
            print(self.scorer_2, ':', results_2)

    # TODO write all necessary params to log file

if __name__ == "__main__":
    dataset_dirs = [
        "../../../data/pan16-author-diarization-training-dataset-problem-a-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-b-2016-02-16",
        "../../../data/pan16-author-diarization-training-dataset-problem-c-2016-02-16"
    ]

    features_file_name = 'basic_feature_extraction/features_files/features1m.json'
    params = dict()

    # Change the task here
    task = 'b'

    if task == 'a':
        print("Loading dataset for task ", task, "...")
        params['task'] = task
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[0]).load_dataset()
        params['context_size'] = 16  # 16 for AggD
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = AgglomerativeDiarizer() # GaussianMixtureDiarizer() #IsolationForestDiarizer()  # AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = MicroScorer
        params['scorer_class_2'] = MacroScorer
        params['cacher'] = Cacher(dir='.cache-task-a')
        params['select-model'] = False
        params['train_size'] = 0.5  # 0.5 for Agg
        params['random_state'] = 9

    elif task == 'b':
        print("Loading dataset for task ", task, "...")
        params['task'] = task
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[1]).load_dataset()
        params['context_size'] = 140  # 140 for kmeans
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = AgglomerativeDiarizer() #AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = BCubedScorer
        params['scorer_class_2'] = None
        params['cacher'] = Cacher(dir='.cache-task-b')
        params['select-model'] = True
        params['train_size'] = 0.5  # 0 za agg iz a
        params['random_state'] = 8

    elif task == 'c':
        print("Loading dataset for task ", task, "...")
        params['task'] = task
        params['dataset'] = Pan16DatasetLoader(dataset_dirs[2]).load_dataset()
        params['context_size'] = 140
        params['document_preprocessor'] = TokenizerPreprocessor()
        params['basic_feature_extractor'] = BasicFeatureExtractor(features_file_name)
        params['feature_transformer'] = None
        params['model'] = DBSCANDiarizer()  #AffinityPropDiarizer() #MeanShiftDiarizer() #DBSCANDiarizer()  # AgglomerativeDiarizer()  #KMeansDiarizer()
        params['scorer_class_1'] = BCubedScorer
        params['scorer_class_2'] = None
        params['cacher'] = Cacher(dir='.cache-task-c')
        params['select-model'] = False
        params['train_size'] = 0.1  # 0.1 for DBSCAN
        params['random_state'] = 7  # 7 for DBSCAN

    pipeline = LearningPipeline(params)
    pipeline.do_chain()






