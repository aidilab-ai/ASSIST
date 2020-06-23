import sys
sys.path.insert(1, '../from_scratch/decorators/preprocessors')

from word_context_pairs_generator import WordContextPairsGenerator
from text_preprocessor import TextPreprocessor
from qit_cleaner import QITEmailBodyCleaner
from integer_encoder import IntegerEncoder
from tokenizer import Tokenizer

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


sentences = np.genfromtxt('./tickets_QIT.txt', dtype=str, delimiter='\n')

prep = TextPreprocessor(sentences)
prep = QITEmailBodyCleaner(prep)
prep = Tokenizer(prep, language='italian')
tokens = prep.preprocess()
vocabulary = build_vocabulary(tokens)

unknown_token_id = max(vocabulary.values()) + 1
prep = IntegerEncoder(prep, vocabulary, unknown_token_id)
prep = WordContextPairsGenerator(prep, window_length=2)

word_context_pairs = prep.preprocess()
target_words = [tw for (tw, cw) in word_context_pairs]
context_words = [cw for (tw, cw) in word_context_pairs]

np.savetxt('target_words.txt', target_words, fmt='%d')
np.savetxt('context_words.txt', context_words, fmt='%d')
