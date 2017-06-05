# Inpladesys - Intrinsic Plagiarism Detection System

This repository contains an implementation and a description paper for an author diarization (and intrinsic plagiarism detection) system as a result of a student project which is a part of [*Text Analysis and Retrieval*](https://www.fer.unizg.hr/en/course/taar) (*TAR*) coursework.

## Author Diarization and Intrinsic Plagiarism Detection
Author diarization is the problem of segmenting text within a document into classes each corresponding to an author, i.e. assignment of every part of a text to an author. The analyzed document can be a result of collaborative work or plagiarism by a single author. The problem is in more detail described [here](http://pan.webis.de/clef16/pan16-web/author-identification). The problem can be divided into 3 subproblems:
* **Traditional intrinsic plagiarism detection.** It is assumed that at least 70% of the text is written by a single author and the rest by other authors. The task is to segment the text into 2 classes.
* **Diarization with a given number of authors.** The task is to segment the text into *n* classes, each representing an author.
* **Diarization with an unknown number of authors.** The task is to segment the text into *n* classes, with the number of authors *n* not being known in advance.

## The Assignment (from [here](http://www.fer.unizg.hr/_download/repository/TAR-2017-ProjectTopics.pdf))
Intrinsic plagiarism detection (also called author diarization) attempts to detect plagiarized text fragments within a single document (in contrast with standard plagiarism detection across documents). Given a document, the task is to identify and group text fragments that were written by different authors. To make things a bit harder, the author change may occur at any position in the text, and not only at sentence or paragraph boundaries. The task is split into three subtasks of increasing difficulty, based on the number of (known) authors. First subtasks assumes two, second a fixed number, and third an unknown number of authors.
### [PAN competition website](http://pan.webis.de/clef16/pan16-web/author-identification.html)
