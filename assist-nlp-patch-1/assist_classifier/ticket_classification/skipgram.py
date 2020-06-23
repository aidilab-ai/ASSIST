import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
import numpy as np


class SkipgramModel(object):
	def __init__(self, config):
		self.config = config

	def convertGloveToW2V(self):
		glove2word2vec(self.config.skipgram_path, 'C:/Users/anton/Desktop/Lavoro/NLG/Dataset/word_embedding_antonio/home/word2vec_.txt')

	def get_skipgram(self):
		skipgram_model = Word2Vec.load(self.config.skipgram_path)
		#
		return skipgram_model


	def save_keyed_vectors(self,skipgram_model, out):
		word_vectors = skipgram_model.wv
		#
		word_vectors.save(out)


	def getCustomEmbeddingMatrix(self,skipgram_model, index_to_word):
		embedding_matrix = np.zeros((len(index_to_word), self.config.input_word_emb_size))
		#
		for i in range(len(index_to_word)):
			token = index_to_word[i]
			if token in skipgram_model.wv.vocab:
				embedding_matrix[i] = skipgram_model.wv[token]
		#
		return embedding_matrix