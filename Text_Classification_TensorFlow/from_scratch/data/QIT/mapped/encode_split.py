import sys
sys.path.insert(1, '../../../decorators/preprocessors')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from text_preprocessor import TextPreprocessor
from qit_cleaner import QITEmailBodyCleaner
from integer_encoder import IntegerEncoder
from tokenizer import Tokenizer
from padder import Padder

from collections import Counter
import numpy as np


sentences = np.genfromtxt('../upsampled/x_QIT.txt', delimiter='\n', dtype=str)
language = 'italian'
max_words = None
max_length = 25

# Text preprocessor with no functionalities whatsoever
prep = TextPreprocessor(sentences)

# Add decorator to clean email bodies
prep = QITEmailBodyCleaner(prep)

# Add tokenizer decorator
prep = Tokenizer(prep, language)

# Load vocabulary
with open('vocabulary_wikipedia', 'r') as vocabulary_file:
    vocabulary = eval( vocabulary_file.read() )

# Add integer encoding decorator
unknown_token_id = max(vocabulary.values()) + 1
prep = IntegerEncoder(prep, vocabulary, unknown_token_id)

# Add padding decorator
padding_token_id = max(vocabulary.values()) + 2
prep = Padder(prep, padding_token_id, max_length)

# Get final tokens
final_tokens = prep.preprocess()

# Load labels
labels = np.genfromtxt('../upsampled/y_QIT.txt', delimiter='\n', dtype=str).reshape((-1, 1))

# Convert labels into one-hot dummies
enc = OneHotEncoder(sparse=False)
one_hot_labels = enc.fit_transform(labels)

# Split dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(final_tokens, one_hot_labels, test_size=0.3, stratify=labels)

# Save the resulting training and test data
np.savetxt('../upsampled/x_train_upsampled.txt', x_train, fmt='%d')
np.savetxt('../upsampled/x_test_upsampled.txt', x_test, fmt='%d')
np.savetxt('../upsampled/y_train_upsampled.txt', y_train, fmt='%d')
np.savetxt('../upsampled/y_test_upsampled.txt', y_test, fmt='%d')
