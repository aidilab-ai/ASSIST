from text_classification_model import TextClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
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
parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing a checkpoint of a previously trained model')
parser.add_argument('--checkpoint-dir-finetune', type=str, required=True, help='Directory where a checkpoint of the fine-tuned model will be saved')
args = parser.parse_args()

# data_dir = './data'
# checkpoint_dir = './checkpoints'
# word2vec_model_path = '/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/data/word_embedding_models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m'

# embedding_size = 256
# lstm_units = 64
# batch_size = 128
#
# epochs = 20
# validation_fraction = 0.2
# dropout_keep_prob = 0.9
# learning_rate = 1e-3
# l2_regularization = 1e-4

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

# Initialize the model
model = TextClassifier(training=True,
                       vocabulary_size=vocabulary_size,
                       embedding_size=config['embedding_size'],
                       lstm_units=config['lstm_units'],
                       output_units=labels_train.shape[1],
                       word2vec_model_path=args.word2vec_model_path)

# Test the model
with model.restore(args.checkpoint_dir, recurrent_part_only=True) as session:

        # Fine-tune the model
        model.fit(tickets_train,
                  labels_train,
                  config['batch_size'],
                  config['epochs'],
                  validation_fraction=config['validation_fraction'],
                  dropout_keep_prob=config['dropout_keep_prob'],
                  learning_rate=config['learning_rate'],
                  l2_regularization=config['l2_regularization'],
                  checkpoint_dir=args.checkpoint_dir_finetune,
                  tensorboard_logdir=args.tensorboard_logdir)
