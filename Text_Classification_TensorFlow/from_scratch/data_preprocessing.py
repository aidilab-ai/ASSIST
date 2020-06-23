from sklearn.preprocessing import OneHotEncoder
from tokenizer import IntegerTokenizer
import pandas as pd
import numpy as np
import argparse
import sys
import os

arg_parser = argparse.ArgumentParser(sys.argv[0])
arg_parser.add_argument('-s', '--samples-files', dest='samples_files', required=True, nargs='+', type=str, help='files containing one sentence per line')
arg_parser.add_argument('-l', '--labels-files', dest='labels_files', required=True, nargs='+', type=str, help='files containing one label per line')
arg_parser.add_argument('-v', '--vocabulary', dest='vocabulary', required=True, type=str, help='file containing the vocabulary (as a Python dictionary)')

args = arg_parser.parse_args()

# data_dir = '../data'
# new_data_dir = './data'
max_sequence_length = 25

# Load vocabulary
with open(args.vocabulary, 'r') as vocabulary_file:
    vocabulary_string = vocabulary_file.read()
    vocabulary = eval(vocabulary_string)

print('Vocabulary size: {}'.format(len(vocabulary)))

# Initialize tokenizer
tokenizer = IntegerTokenizer(vocabulary)

# Load training and test data (I'm using pandas instead of numpy because np.loadtxt happens to have a bug that causes it to read more lines than needed)
for samples_file in args.samples_files:

    samples = pd.read_csv(samples_file, dtype=str, delimiter='\n', header=None).values[:, 0]
    sequences = tokenizer.tokenize_and_pad(samples, sequence_length=max_sequence_length)
    np.savetxt('{}.tokenized'.format(samples_file), sequences, fmt='%d')


encoder = OneHotEncoder(sparse=False)

# Load labels
for labels_file in args.labels_files:

    labels = pd.read_csv(labels_file, dtype=str, delimiter='\n', header=None).values
    labels_one_hot = encoder.fit_transform(labels)
    np.savetxt('{}.onehot'.format(labels_file), labels_one_hot, fmt='%d')

# x_train = pd.read_csv(os.path.join(data_dir, 'x_train.txt'), dtype=str, delimiter='\n', header=None).values[:, 0]
# y_train = pd.read_csv(os.path.join(data_dir, 'y_train.txt'), dtype=str, delimiter='\n', header=None).values[:, 0]
# x_test = pd.read_csv(os.path.join(data_dir, 'x_test.txt'), dtype=str, delimiter='\n', header=None).values[:, 0]
# y_test = pd.read_csv(os.path.join(data_dir, 'y_test.txt'), dtype=str, delimiter='\n', header=None).values[:, 0]
#
#
# tickets_train = tokenizer.tokenize_and_pad(x_train, sequence_length=max_sequence_length)
# tickets_test = tokenizer.tokenize_and_pad(x_test, sequence_length=max_sequence_length)
#
#
# labels_train = encoder.fit_transform( y_train.reshape(-1, 1) )
# labels_test = encoder.fit_transform( y_test.reshape(-1, 1) )
#
# np.savetxt(os.path.join(new_data_dir, 'x_train.txt'), tickets_train, fmt='%d')
# np.savetxt(os.path.join(new_data_dir, 'y_train.txt'), labels_train, fmt='%d')
# np.savetxt(os.path.join(new_data_dir, 'x_test.txt'), tickets_test, fmt='%d')
# np.savetxt(os.path.join(new_data_dir, 'y_test.txt'), labels_test, fmt='%d')
