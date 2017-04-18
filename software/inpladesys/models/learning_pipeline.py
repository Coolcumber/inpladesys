

class LearningPipeline:

    def __init__(self, parameters: tuple):
        self.dataset = parameters['dataset']
        self.document_preprocessor = parameters['document_preprocessor']
        self.feature_extractor = parameters['feature_extracotr']
        self.feature_postprocessor = parameters['feature_postprocessor']
        self.model = parameters['model']

    def do_chain(self):  # TODO split dataset into train, validation and test set if needed
        for i in range(self.dataset.size):
            document, segmentation = self.dataset[i]
            preprocessed_doc = self.document_preprocessor.preprocess(document)
            self.feature_extractor.extract_features(preprocessed_doc)








