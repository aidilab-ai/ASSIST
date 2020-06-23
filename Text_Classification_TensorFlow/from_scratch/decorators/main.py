import sys
sys.path.insert(1, './preprocessors')

from text_preprocessor import TextPreprocessor
from qit_cleaner import QITEmailBodyCleaner
from integer_encoder import IntegerEncoder
from tokenizer import Tokenizer
from padder import Padder

from collections import Counter
import numpy as np


def build_vocabulary(sequences, max_words=None):

    words = []
    for token_sequence in sequences:
        words.extend(token_sequence)

    word_counts = dict( Counter(words).most_common(max_words) )

    most_common_words = list( word_counts.keys() )
    word_ids = list( range( len(most_common_words) ) )

    vocabulary = dict( zip(most_common_words, word_ids) )
    return vocabulary


sentences = np.genfromtxt('./tickets_QIT.txt', delimiter='\n', dtype=str)
language = 'italian'
max_words = None
max_length = 30

# Text preprocessor with no functionalities whatsoever
prep = TextPreprocessor(sentences)

# Add decorator to clean email bodies
prep = QITEmailBodyCleaner(prep)

# Add tokenizer decorator
prep = Tokenizer(prep, language)

# Get intermediate results
tokens = prep.preprocess()

# Build vocabulary
vocabulary = build_vocabulary(tokens, max_words=max_words)

# Add integer encoding decorator
unknown_token_id = max(vocabulary.values()) + 1
prep = IntegerEncoder(prep, vocabulary, unknown_token_id)

# Add padding decorator
padding_token_id = max(vocabulary.values()) + 2
prep = Padder(prep, padding_token_id, max_length)

# Get final tokens
final_tokens = prep.preprocess()

# Save the resulting dataset
np.savetxt('tokenized_tickets.txt', final_tokens, fmt='%d')
