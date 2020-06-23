# ASSIST - Text Classification on support requests

The directory `assist-nlp-patch-1` contains the original code written by Quest-it.

The directory `assist-modified` contains some modifications I made to the original code.

The directory `Text_Classification_TensorFlow/from_scratch` contains the code I have written from scratch for training a text classification model for Ambrogio's support requests.

The directory `Text_Classification_TensorFlow/Word2Vec` contains the code adapted from an old TensorFlow v1 tutorial about training word embeddings from scratch. In practice, it is convenient to use [pre-trained word embeddings](http://hlt.isti.cnr.it/wordembeddings/), especially when you don't have a large dataset (like in the case of Ambrogio).

**Note**: This code doesn't work out of the box because pre-trained models and embeddings are missing in this repository (their total size was ~12GB)
