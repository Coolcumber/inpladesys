# + TODO allow SimpleFixedAuthorDiarizer to adjust number of authors in BASELINE tests
# + TODO implement scorers as parameters for learning pipeline ?
# + TODO implement interfacce for models
# TODO fix labeling of non-clustered examples
# TODO baseline results which use rand should be evaluated on 50 runs
# + TODO fix warnings in training
# + TODO add new features (new "iteration"):
#   + type-token ratio (vocabulary richness)
#   - word n-grams ?
#   + char n-grams of POS tags
#   - chunks (phrases)
#   - synonyms ?
#   - lexical density: the ratio of content words to grammatical words
#   - LSA/LSI/word2vec udaljenosti trenutnog tokena od ostalih u windowu
# TODO try other algorithms: DBSCAN, OPTICS, HAC, Spectral, EM
# TODO add writing of test results to a file
# + TODO experiment with params (what combs ?)
# TODO solve task a
# TODO finish neural net
# TODO implement elbow method (or some other ?) for task c
# TODO implement t-test for all tasks
# task a: window size is cca. 14-18
# + TODO leave data sparse in basic_feature_extractor - not
# TODO move scaling in diarizers to AbstractDiarizer

