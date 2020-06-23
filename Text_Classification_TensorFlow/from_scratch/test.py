from text_classification_model import TextClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument('--test-features', type=str, required=True, help='Test features (one sample per line, features separated by whitespaces)')
parser.add_argument('--test-targets', type=str, required=True, help='Test targets (one sample per line, one-hot encoded)')
parser.add_argument('--config-file-path', type=str, required=True, help='Path to a configuration file')
parser.add_argument('--vocabulary', type=str, required=True, help='Vocabulary file (sintax of a Python dictionary mapping words to integers)')
parser.add_argument('--word2vec-model-path', type=str, required=False, help='Path to a pre-trained word2vec model')
parser.add_argument('--checkpoint-dir', type=str, required=False, help='Directory containing a checkpoint of a previously trained model')
args = parser.parse_args()

# data_dir = './data'
# checkpoint_dir = './checkpoints'
# word2vec_model_path = '/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/data/word_embedding_models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m'

# embedding_size = 256
# lstm_units = 64
# batch_size = 128

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
tickets_test = pd.read_csv(args.test_features, dtype=int, delimiter=' ').values
labels_test = pd.read_csv(args.test_targets, dtype=int, delimiter=' ').values

# Initialize the model
model = TextClassifier(training=False,
                       vocabulary_size=vocabulary_size,
                       embedding_size=config['embedding_size'],
                       lstm_units=config['lstm_units'],
                       output_units=labels_test.shape[1],
                       word2vec_model_path=args.word2vec_model_path)

# Test the model
with model.restore(args.checkpoint_dir) as session:
    logits = model.predict(session, tickets_test, labels_test, config['batch_size'])

y_true = np.argmax(labels_test, 1)
y_pred = np.argmax(logits, 1)

print('Test accuracy: {:.2f}%'.format( np.mean(y_pred == y_true) * 100 ))

print('Confusion matrix:')
print( confusion_matrix(y_true, y_pred) )
