import numpy as np
import json

"""##################################################################"""
"""Config"""
"""##################################################################"""
class Config():
	def __init__(self, vocab_size=70000, test=False, main_path="/home/questit/data/models_and_data/"):
		#
		self.is_test = test
		self.transfer_learning = True
		#
		self.device = "CPU"
		self.device_count = {"CPU": 4}
		# Log
		self.verbose = "high"
		self.tensorboard_saving = False
		# Data
		self.op_encoding = "windows-1252"
		# self.csv_encoding = "latin-1"
		# self.csv_encoding = "ISO-8859-1"
		self.csv_encoding = "UTF-8"
		self.csv_quote = None
		self.csv_delimiter = "|"
		self.num_classes = 5
		#Clean and Balancing Data
		self.splitDataTrainingTest = False
		self.use_remove_identical_ticket = False
		self.use_balancing_data = True
		self.balancing_cut = 25000
		self.balancing_classes = dict([('Generico', 12), ('Magazzino', 18), ('Commerciale', 12)])
		self.labels = ["Generico","Amministrazione","Tecnico","Commerciale","Magazzino"]
		self.labels_map = {}
		self.max_num_tickets = 250000
		self.max_model_checkpoints = 3
		#Data paths
		self.main_path = main_path
		#self.main_path = "/home/rigutini/ticket_classification/"
		# self.main_path = "C:/Users/anton/Desktop/ticket_classification_Eliza_20181107_epochs_200/"
		self.csv_path = self.main_path + "onlyApertura/SupportRequests.txt"
		self.data_path = self.main_path + "onlyApertura/"
		self.model_path = self.main_path + "saved_model/"
		self.tensorboard_path = self.main_path + "tensorboard/"
		self.best_model_path = self.main_path + "best_models/"
		self.data_sequences_path = self.main_path + "parsed_sequences/"
		# self.model_path = "C:/Users/anton/Desktop/ticket_classification_Eliza_20181107_epochs_200/saved_model/"
		self.model_to_restore = "model.ckpt-3.meta"
		self.training_result_path = self.main_path + "training_results/"

		#Vocabulary
		self.loadOrbuild_dictionary = "load"
		self.vocab_size = vocab_size
		self.unkown_token = '<UNK>'
		self.pad_token = '<PAD>'
		self.numeric_token = '<NUMERIC>'
		self.date_token = '<DATE>'
		self.currency_token = '<CURRENCY>'
		#remove tickets with token smaller than the given number
		self.threshold_len=5

		# Feature Extractor
		self.features_to_extract = ["check_isINVoc", "check_isNumeric", "check_isCurrency", "check_isDate"]

		#Neural Network
		#Input
		self.batch_size = 8
		self.epochs = 300
		self.max_length_sequence = 30
		#Word Embedding
		self.use_pretrained_embs = True
		self.input_word_emb_size = 300
		self.skipgram_path = '/home/questit/data/word_embedding_models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m'
		#self.skipgram_path = 'C:/Users/anton/Desktop/Lavoro/NLG/Dataset/word_embedding_antonio/home/berardi/glove_WIKI'
		#self.skipgram_path = '/home/rigutini/sequence2sequence/skipgram_model/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m'
		self.skipgramEmbedding = np.zeros((self.batch_size, self.input_word_emb_size), dtype=float)
		self.use_embedding_dropout = False
		self.embedding_dropout_keep_prob = 0.7
		#Encoder LSTM
		self.Encoder_type  = "LSTM"
		self.encoder_rnn_size = 256
		# MLP
		self.hidden_units = 128
		self.dropout_use = True
		self.dropout_keep_prob = 0.8
		self.numb_layers = 1
		#Regularization
		self.regularization_use = True
		self.regularization_type = "L2"
		self.regularization_beta = 0.01

		# Optimizer
		self.optimizer_type  = "Adam"
		self.lr = 0.001

		#cnn model
		self.filter_sizes = "3,4,5"
		self.num_filters = 128
		self.dropout_keep_prob = 0.5
		self.l2_reg_lambda = 0.0

	"""##########################################################################"""

	def loadConfigFile(self,filePath):
		with open(filePath) as f:
			data = json.load(f)
		return data

	"""##########################################################################"""

	def mapMain2Config(self, dataJson):
		main = dataJson["main"]
		self.is_test = main["is_test"]
		self.transfer_learning = main["transfer_learning"]
		self.device = main["device"]
		self.device_count = main["device_count"]
		self.verbose = main["verbose"]
		self.tensorboard_saving = main["tensorboard_saving"]
		self.max_model_checkpoints = main["max_model_checkpoints"]

	"""##########################################################################"""

	def mapData2Config(self, dataJson):
		data = dataJson["data"]
		self.op_encoding = data["op_encoding"]
		self.csv_encoding = data["csv_encoding"]
		self.csv_quote = data["csv_quote"]
		self.csv_delimiter = data["csv_delimiter"]
		self.num_classes = data["num_classes"]
		self.labels = data["labels"]
		self.labels_map = data["labels_map"]
		self.max_num_tickets = data["max_num_tickets"]

	"""##########################################################################"""

	def mapCleanBalancing2Config(self, dataJson):
		data = dataJson["clean_balancing_data"]
		self.splitDataTrainingTest = data["splitDataTrainingTest"],
		self.use_remove_identical_ticket = data ["use_remove_identical_ticket"],
		self.use_balancing_data = data["use_balancing_data"],
		self.balancing_cut = data["balancing_cut"],
		self.balancing_classes = data["balancing_classes"]

	"""##########################################################################"""

	def mapPaths2Config(self, dataJson):
		data = dataJson["paths"]
		self.main_path = data["main_path"]
		self.csv_path = self.main_path + data["csv_path"]
		self.data_path = self.main_path + data["data_path"]
		self.model_path = self.main_path + data["model_path"]
		self.tensorboard_path = self.main_path + data["tensorboard_path"]
		self.best_model_path = self.main_path + data["best_model_path"]
		self.data_sequences_path = self.main_path + data["data_sequences_path"]
		self.model_to_restore = data["model_to_restore"]
		self.training_result_path = self.main_path + data["training_result_path"]

	"""##########################################################################"""

	def mapVocabulary2Config(self, dataJson):
		data = dataJson["vocabulary"]
		self.loadOrbuild_dictionary = data["loadOrbuild_dictionary"]
		self.vocab_size = data["vocab_size"]
		self.unkown_token = data["unkown_token"]
		self.pad_token = data["pad_token"]
		self.numeric_token = data["numeric_token"]
		self.date_token = data["date_token"]
		self.currency_token = data["currency_token"]
		# remove tickets with token smaller than the given number
		self.threshold_len = 5

	"""##########################################################################"""

	def mapFeatureExtractor2Config(self, dataJson):
		data = dataJson["feature_extractor"]
		self.features_to_extract = data["features_to_extract"]

	"""##########################################################################"""

	def mapNeuralNetwork2Config(self, dataJson):
		data = dataJson["neural_network"]
		# Input
		input = data["input"]
		self.batch_size = input["batch_size"]
		self.epochs = input["epochs"]
		self.max_length_sequence = input["max_length_sequence"]
		# Word Embedding
		word_embedding = data["word_embedding"]
		self.use_pretrained_embs = word_embedding["use_pretrained_embs"]
		self.input_word_emb_size = word_embedding["input_word_emb_size"]
		self.skipgram_path = word_embedding["skipgram_path"]
		self.skipgramEmbedding = np.zeros((self.batch_size, self.input_word_emb_size), dtype=float)
		self.use_embedding_dropout = word_embedding["use_embedding_dropout"]
		self.embedding_dropout_keep_prob = word_embedding["embedding_dropout_keep_prob"]
		# Encoder LSTM
		lstm = data["lstm"]
		self.Encoder_type = lstm["encoder_type"]
		self.encoder_rnn_size = lstm["encoder_rnn_size"]
		# MLP
		mlp = data["mlp"]
		self.hidden_units = mlp["hidden_units"]
		self.dropout_use = mlp["dropout_use"]
		self.dropout_keep_prob = mlp["dropout_keep_prob"]
		self.numb_layers = mlp["numb_layers"]
		# Regularization
		regularization = data["regularization"]
		self.regularization_use = regularization["regularization_use"]
		self.regularization_type = regularization["regularization_type"]
		self.regularization_beta = regularization["regularization_beta"]

		# Optimizer
		optimizer = data["optimizer"]
		self.optimizer_type = optimizer["optimizer_type"]
		self.lr = optimizer["lr"]

	"""##########################################################################"""

	def configFromFile(self, dataPath):
		data = self.loadConfigFile(dataPath)
		self.mapMain2Config(data)
		self.mapData2Config(data)
		self.mapCleanBalancing2Config(data)
		self.mapFeatureExtractor2Config(data)
		self.mapPaths2Config(data)
		self.mapNeuralNetwork2Config(data)
		self.mapVocabulary2Config(data)

	"""##########################################################################"""
	def updateDataOfMainPath(self,dataJson, mainPath):
		data = self.loadConfigFile(dataJson)
		data = data["paths"]
		self.main_path = mainPath
		self.csv_path = self.main_path + data["csv_path"]
		self.data_path = self.main_path + data["data_path"]
		self.model_path = self.main_path + data["model_path"]
		self.tensorboard_path = self.main_path + data["tensorboard_path"]
		self.best_model_path = self.main_path + data["best_model_path"]
		self.data_sequences_path = self.main_path + data["data_sequences_path"]
		self.model_to_restore = data["model_to_restore"]
		self.training_result_path = self.main_path + data["training_result_path"]
