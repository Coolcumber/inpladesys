# Inpladesys - Intrinsic Plagiarism Detection System

This repository is going to contain an implementation and a description paper for an author diarization (and intrinsic plagiarism detection) system as a result of a student project which is a part of [*Text Analysis and Retrieval*](https://www.fer.unizg.hr/en/course/taar) (*TAR*) coursework.

## Author Diarization and Intrinsic Plagiarism Detection
Author diarization is the problem of segmenting text within a document into classes each corresponding to an author, i.e. assignment of every part of a text to an author. The analyzed document can be a result of collaborative work or plagiarism by a single author. The problem is more thoroughly described [here](http://pan.webis.de/clef16/pan16-web/author-identification). The problem can be divided into 3 subproblems:
* **Traditional intrinsic plagiarism detection.** It is assumed that at least 70% of the text is written by a single author and the rest by other authors. The task is to segment the text into 2 classes.
* **Diarization with a given number of authors.** The task is to segment the text into *n* classes, each representing an author.
* **Diarization with an unknown number of authors.** The task is to segment the text into *n* classes, with the number of authors *n* not being known in advance.

## The Assignment ([source](http://www.fer.unizg.hr/_download/repository/TAR-2017-ProjectTopics.pdf))
Intrinsic plagiarism detection (also called author diarization) attempts to detect plagiarized text fragments within a single document (in contrast with standard plagiarism detection across documents). Given a document, the task is to identify and group text fragments that were written by different authors. To make things a bit harder, the author change may occur at any position in the text, and not only at sentence or paragraph boundaries. The task is split into three subtasks of increasing difficulty, based on the number of (known) authors. First subtasks assumes two, second a fixed number, and third an unknown number of authors.
### [PAN competition website](http://pan.webis.de/clef16/pan16-web/author-identification.html)
### Entry points:
* [Paolo Rosso, Francisco Rangel, Martin Potthast, Efstathios Stamatatos, Michael Tschuggnall, and Benno Stein. Overview of PAN’16 - New Challenges for Authorship Analysis: Cross-genre Profiling, Clustering, Diarization, and Obfuscation.](http://www.uni-weimar.de/medien/webis/publications/papers/stein_2016i.pdf)
* [Mikhail Kuznetsov, Anastasia Motrenko, Rita Kuznetsova, and Vadim Strijov. Methods for Intrinsic Plagiarism Detection and Author Diarization.](http://www.uni-weimar.de/medien/webis/events/pan-16/pan16-papers-final/pan16-author-identification/kuznetsov16-notebook.pdf)
* [Abdul Sittar, Hafiz Rizwan Iqbal, and Rao Muhammad Adeel Nawab. Author Diarization Using Cluster-Distance Approach.](http://www.uni-weimar.de/medien/webis/events/pan-16/pan16-papers-final/pan16-author-identification/sittar16-notebook.pdf)

## [Project Plan](https://drive.google.com/drive/folders/0BzQ2SbanL1zCa1VoSVJBLXBxUXM)
### Milestones
Beside the checkpoints defined by the course, we have defined a set of informal checkpoints to ensure that all the project tasks are done on time. These checkpoints are *italicized*.
#### 2017-3-24 Project plan submission
#### 2017-4-8 *Baseline checkpoint*
* at least a working baseline for each subtask
* working evaluation
#### 2017-5-1 *Kuznetsov checkpoint*
* a *well-working* model
#### 2017-5-8 Project checkpoint
* final model(s)
#### 2017-5-19 *Paper checkpoint*
* final paper ready for reviews
#### 2017-6-9 Project presentations
#### 2017-6-18 Project reviews
#### 2017-6-26 Final papers

### Models
#### Feature set
* *feature-set-1* - a small set of simple lexical features **TODO**
#### Feature preprocessing
* *normalized* - translated to 0-mean and scaled to unit variance
* *t1* - weighted by minimizing (*variance within class averaged across all classes*) - *λ*(*variance among class-centroids*), where *λ* is a scalar hyperparameter (the problem is described [here](http://mathb.in/134812))
#### Baseline 1
* the whole document is predicted to have been written by a single author
#### Simple model
* feature set: *feature-set-1*
* feature preprocessing: *normalized*
* clustering/classification:
  * the length of segments is predefined: 7 words
  * Euclidean distance is used
  * *k-means*(*++*) clustering (with a predefined number of authors (*n*=3) for task 3)
#### Simple model with transformed features
* everything as in *Simple model* except:
  * feature preprocessing: *t1*
