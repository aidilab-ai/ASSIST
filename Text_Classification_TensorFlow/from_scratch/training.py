from text_classification_model import TextClassifier
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import itertools
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument('--training-features', type=str, required=True, help='Training features (one sample per line, features separated by whitespaces)')
parser.add_argument('--training-targets', type=str, required=True, help='Training targets (one sample per line, one-hot encoded)')
parser.add_argument('--vocabulary', type=str, required=True, help='Vocabulary file (sintax of a Python dictionary mapping words to integers)')
parser.add_argument('--config-file-path', type=str, required=True, help='Path to a configuration file')
parser.add_argument('--tensorboard-logdir', type=str, required=True, help='Directory where TensorBoard summaries will be saved')
parser.add_argument('--word2vec-model-path', type=str, required=False, help='Path to a pre-trained word2vec model')
parser.add_argument('--checkpoint-dir', type=str, required=False, help='Directory where you wish to save a checkpoint of the trained model')
args = parser.parse_args()

# data_dir = './data'
# checkpoint_dir = './checkpoints'
# word2vec_model_path = '/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/data/word_embedding_models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m'
# tensorboard_logdir = '/home/anfri/Lavoro-AiDiLab/ASSIST/Text_Classification_TensorFlow/from_scratch/tensorboard'

# embedding_size = 256 # Doesn't count if you use pre-trained word embeddings
# lstm_units = [32, 64, 128]
# batch_sizes = [128, 512]
# epochs = 20
# validation_fraction = 0.2
# dropout_keep_probs = [0.7, 0.8, 0.9]
# learning_rates = [1e-2, 1e-3, 1e-4]
# l2_regularizations = [1e-3, 1e-4, 1e-5]

# Open configuration file and build a dictionary off of it
with open(args.config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Load vocabulary
with open(args.vocabulary, 'r') as vocabulary_file:
    vocabulary_string = vocabulary_file.read()
    vocabulary = eval(vocabulary_string)

vocabulary_size = len(vocabulary)
del vocabulary

# Load data
tickets_train = pd.read_csv(args.training_features, dtype=int, delimiter=' ').values
labels_train = pd.read_csv(args.training_targets, dtype=int, delimiter=' ').values

# Perform a grid search over all the possible hyperparameters
for lu, bs, kp, lr, reg in itertools.product(config['lstm_units'],
                                             config['batch_sizes'],
                                             config['dropout_keep_probs'],
                                             config['learning_rates'],
                                             config['l2_regularizations']):

    # Needed to avoid problems concerning the reusing of variables
    tf.reset_default_graph()

    # Initialize the model
    model = TextClassifier(training=True,
                           vocabulary_size=vocabulary_size,
                           embedding_size=config['embedding_size'],
                           lstm_units=lu,
                           output_units=labels_train.shape[1],
                           word2vec_model_path=args.word2vec_model_path)

    # Train the model
    model.fit(tickets_train,
              labels_train,
              bs,
              config['epochs'],
              validation_fraction=config['validation_fraction'],
              dropout_keep_prob=kp,
              learning_rate=lr,
              l2_regularization=reg,
              checkpoint_dir=args.checkpoint_dir,
              tensorboard_logdir=args.tensorboard_logdir)

    del model
