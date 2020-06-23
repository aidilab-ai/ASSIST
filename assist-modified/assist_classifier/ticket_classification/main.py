import re
import os
import tokenizer as tk
import vocabulary as vc
import data as dt
import config as cg
import utility as ut
import model as ml
import filter_data as fd
import skipgram as sk
import tensorflow as tf

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

"""##################################################################"""
"""Initialization"""
def initialization(config):
	print("Initialization ... check directories")
	#check that Main and Data Path exists!
	if not os.path.exists(config.main_path):
		raise Exception("Main Path : " + str(config.main_path) + " does not exist!")
	if not os.path.exists(config.data_path):
		raise Exception("Data Path : " + str(config.data_path) + " does not exist!")

	paths_to_check = [config.model_path, config.best_model_path, config.tensorboard_path, config.data_sequences_path]
	for path in paths_to_check:
		exist = ut.checkDircetoryExistence(path)
		if not exist:
			ut.createDirectory(path)


"""##################################################################"""

def cleanData():
	config = cg.Config()
	dataL = dt.Data(config)
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	tickets = dataL.loadDataInArray(config.main_path + "onlyApertura20181129/tickets.txt", config.csv_encoding)
	targets = dataL.loadDataInArray(config.main_path + "onlyApertura20181129/targets_mapped.txt")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	tickets, targets = ut.removeIdenticalTickets(tickets, targets)
	dataL.writeArrayInFile(tickets, 'onlyApertura20181129/tickets_identicalRemoved.txt', "utf-8")
	dataL.writeArrayInFile(targets, 'onlyApertura20181129/targets_identicalRemoved.txt', "utf-8")

#cleanData()

"""##################################################################"""
def preprocessData(tickets,targets,labels):
	#Tokenize data
	tok = tk.Tokenizer(tickets)
	tok.tokenizeTickets()
	tickets_to_lower = tok.toLower()
	tickets_no_sw, targets_no_sw = tok.removeStopWords(tickets_to_lower,targets)
	#create the array of words from all the tickets
	words = tok.extractWords()

	return tickets_no_sw, targets_no_sw, words

"""##################################################################"""

def balanceData(config,tickets, targets, labels):
	dataL = dt.Data(config)
	print("	*** Cutting class dimension at  " + str(config.balancing_cut) + " examples \n")
	tickets, targets = ut.balanceSubSamplesReturnArray(tickets, targets, labels, max_number=config.balancing_cut)
	classes = config.balancing_classes
	for class2b, mult in classes.items():
		print("	*** Balancing class " + str(class2b) + " \n")
		tickets_train, targets_train = dataL.overSampleData(tickets, targets, labels, class2b, mult)
	return tickets_train, targets_train

"""##################################################################"""

def loadAndSplit():
	"""Load the data cleaned."""
	config = cg.Config()
	dataL = dt.Data(config)
	print("	*** Data Loading ... \n")
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	if config.splitDataTrainingTest == True:
		tickets_train = dataL.loadDataInArray(config.data_path + "/tickets.txt", config.csv_encoding)
		targets_train = dataL.loadDataInArray(config.data_path + "/targets.txt")

		print("	*** Data Loaded \n")
		if config.use_remove_identical_ticket == True:
			print("	*** Removing Identical Tickets \n")
			tickets_train, targets_train = ut.removeIdenticalTickets(tickets_train, targets_train)
			dataL.writeArrayInFileCompleteDataPath(tickets_train, config.data_path + '/tickets_identicalRemoved.txt', "utf-8")
			dataL.writeArrayInFileCompleteDataPath(targets_train, config.data_path + '/targets_identicalRemoved.txt', "utf-8")

		tickets_train, tickets_test, targets_train, targets_test = ut.get_train_and_test(targets_train, targets_train, test_size=0.2)
	else:
		print("	*** Load Training and Testing Data \n")
		tickets_train = dataL.loadDataInArray(config.data_path + "/tickets_training.txt", config.csv_encoding)
		targets_train = dataL.loadDataInArray(config.data_path + "/targets_training.txt")
		tickets_test = dataL.loadDataInArray(config.data_path + "/tickets_test.txt", config.csv_encoding)
		targets_test = dataL.loadDataInArray(config.data_path + "/targets_test.txt")

	if config.use_balancing_data == True:
		print("	*** Balancing Training Data \n")
		tickets_train, targets_train = balanceData(config, tickets_train, targets_train, labels)

	dataL.countClassOccurrences(targets_train,labels, "Training")
	dataL.countClassOccurrences(targets_test, labels, "Test")
	if config.use_balancing_data == True:
		print("	*** Saving Balanced Data\n")
		dataL.writeArrayInFileCompleteDataPath(tickets_train, config.data_path +'/tickets_training_balanced.txt', "utf-8")
		dataL.writeArrayInFileCompleteDataPath(targets_train, config.data_path + '/targets_training_balanced.txt', "utf-8")
	#dataL.writeArrayInFileCompleteDataPath(tickets_test, config.data_path + '/tickets_test.txt', "utf-8")
	#dataL.writeArrayInFileCompleteDataPath(targets_test, config.data_path + '/targets_test.txt', "utf-8")

	return tickets_train, tickets_test, targets_train, targets_test

"""##################################################################"""

def loadData():
	"""Load data for the first time. Read the DB dump and create the data based on the 'Aperture'"""
	config = cg.Config()
	dataL = dt.Data(config)
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	# Load Data
	targets, tickets, complete_labels, first_level_targets, first_level_labels = dataL.load_data()
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	map_labels = dataL.loadMapFromJson("C:/Users/anton/Desktop/Lavoro/Ticket_Classification/map_labels.json")
	filtdata = fd.FilterData(config, labels)
	#Filter Data
	tickets, targets = filtdata.filterOutTickets(tickets,targets)
	#tickets, targets = filtdata.filterOutTicketsWithTargetsMap(map_labels['map'], targets, tickets)
	#Map Data
	new_targets = dataL.mapTargets(map_labels['map'], targets)
	tt_complete = []
	count = 0
	for old in targets:
		tt_complete.append(old + " -- > " + new_targets[count])
		count += 1
	#level_zero_map_label = ["Generico", "Amministrazione", "Tecnico", "Commerciale", "Magazzino"]
	dataL.writeArrayInFile(tickets, 'onlyApertura/tickets.txt', "utf-8")
	dataL.writeArrayInFile(targets, 'onlyApertura/complete_targets.txt', "utf-8")
	dataL.writeArrayInFile(new_targets, 'onlyApertura/targets_mapped.txt', "utf-8")
	dataL.writeArrayInFile(tt_complete, 'onlyApertura/tt_multilevel.txt', "utf-8")

#loadData()

"""##################################################################"""

def mainTrainModel():
	print("============ Start =============\n")

	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets\n")
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	tickets = dataL.loadDataInArray(config.data_path + "tickets_balanced_15000.txt", config.csv_encoding)
	targets = dataL.loadDataInArray(config.data_path + "target_balanced_15000.txt")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	print("3 - Preprocess Data\n")
	tickets, targets = ut.removeIdenticalTickets(tickets,targets)
	tickets_to_lower, targets, words = preprocessData(tickets,targets,labels)
	print("4 - Build Vocabulary\n")
	# Create Vocabulary
	voc = vc.Vocabulary(config)
	dictionary, reverse_dict = voc.build_dictionary(words, labels)
	voc.saveDictionary(dictionary,"vocabulary")
	print("5 - Create Ticket Sequences and Targets Hot Vectors\n")
	#Create Sequences and HotVectors for the Target
	tickets_sequences = dataL.createDataSequence(tickets_to_lower,dictionary)
	oneHotVectorTarget = dataL.transformInOneHotVector(labels,targets)
	print("6 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_sequences, oneHotVectorTarget, trash = filtdata.removeTokenOOV(tickets_sequences, oneHotVectorTarget, dictionary)
	print("	*** Class Trash len : " + str(len(trash)))
	print("7 - Generate Training and Testing Dataset\n")
	X_train, X_test, y_train, y_test = ut.get_train_and_test(tickets_sequences, oneHotVectorTarget, test_size=0.2)
	dataL.writeArrayStringInFile(X_train, 'parsed_sequences_15000/tickets_training.txt', "utf-8")
	dataL.writeArrayStringInFile(X_test, 'parsed_sequences_15000/tickets_test.txt', "utf-8")
	dataL.writeArrayStringInFile(y_train, 'parsed_sequences_15000/target_training.txt', "utf-8")
	dataL.writeArrayStringInFile(y_test, 'parsed_sequences_15000/target_test.txt', "utf-8")
	print("	*** Training Size : " + str(len(X_train)) + "\n")
	if config.use_pretrained_embs:
		print("	*** Uso pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel,reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	print("8 - Start Training\n")
	ml.runTraining(config, X_train, y_train, labels)
	print("============ End =============\n")

"""##################################################################"""

def mainTrainModelOnAperture():
	print("============ Start =============\n")

	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets\n")
	tickets_training, tickets_test, targets_training, targets_test = loadAndSplit()

	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	print("3 - Preprocess Data\n")
	tickets_training_tl, targets_training, words = preprocessData(tickets_training,targets_training,labels)
	tickets_test_tl, targets_test, w_ = preprocessData(tickets_test, targets_test, labels)
	if config.loadOrbuild_dictionary == "build":
		print("4 - Build Vocabulary\n")
		# Create Vocabulary
		voc = vc.Vocabulary(config)
		dictionary, reverse_dict = voc.build_dictionary(words, labels)
		voc.saveDictionary(dictionary,"vocabulary")
		print("*** Vocabulary saved \n")
	else:
		print("4 - Load Vocabulary\n")
		#Load Existing Vocabulary
		voc = vc.Vocabulary(config)
		dictionary = voc.loadDictionary("vocabulary")
		reverse_dict = voc.getReverseDictionary(dictionary)

	print("5 - Create Ticket Sequences and Targets Hot Vectors\n")
	#Create Sequences and HotVectors for the Target
	tickets_training_sequences = dataL.createDataSequence(tickets_training_tl,dictionary)
	oneHotVectorTarget_training = dataL.transformInOneHotVector(labels,targets_training)
	tickets_test_sequences = dataL.createDataSequence(tickets_test_tl, dictionary)
	oneHotVectorTarget_test = dataL.transformInOneHotVector(labels, targets_test)
	print("6 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_training_sequences, oneHotVectorTarget_training, trash = filtdata.removeTokenOOV(tickets_training_sequences, oneHotVectorTarget_training, dictionary)
	print("	*** Classe Cestino in Training : " + str(len(trash)) + "\n")
	tickets_test_sequences, oneHotVectorTarget_test, trash = filtdata.removeTokenOOV(tickets_test_sequences,oneHotVectorTarget_test,dictionary)
	print("	*** Classe Cestino in Test : " + str(len(trash)) + "\n")
	print("7 - Generate Training and Testing Dataset\n")
	dataL.writeArrayInFileCompleteDataPath(tickets_training_sequences, config.data_sequences_path + '/tickets_training.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(tickets_test_sequences, config.data_sequences_path + '/tickets_test.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(oneHotVectorTarget_training, config.data_sequences_path + '/target_training.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(oneHotVectorTarget_test, config.data_sequences_path + '/target_test.txt', "utf-8")
	print("	*** Training Size : " + str(len(tickets_training_sequences)) + "\n")
	print("	*** Test Size : " + str(len(tickets_test_sequences)) + "\n")
	if config.use_pretrained_embs:
		print("	*** Use pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel,reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	print("8 - Start Training\n")
	ml.runTraining(config, tickets_training_sequences, oneHotVectorTarget_training, labels)
	print("============ End =============\n")

"""##################################################################"""

def mainTrainModelOnApertureWithSequenceFeatures():
	print("============ Start =============\n")

	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets\n")
	tickets_training, tickets_test, targets_training, targets_test = loadAndSplit()
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	print("3 - Preprocess Data\n")
	tickets_training_tl, targets_training, words = preprocessData(tickets_training,targets_training,labels)
	tickets_test_tl, targets_test, w_ = preprocessData(tickets_test, targets_test, labels)
	print("4 - Build Vocabulary\n")
	# Create Vocabulary
	voc = vc.Vocabulary(config)
	dictionary, reverse_dict = voc.build_dictionary(words, labels)
	voc.saveDictionary(dictionary,"vocabulary")
	print("5 - Create Ticket Sequences and Targets Hot Vectors\n")
	#Create Sequences and HotVectors for the Target
	tickets_training_sequences = dataL.createDataSequence(tickets_training_tl,dictionary)
	oneHotVectorTarget_training = dataL.transformInOneHotVector(labels,targets_training)
	tickets_test_sequences = dataL.createDataSequence(tickets_test_tl, dictionary)
	oneHotVectorTarget_test = dataL.transformInOneHotVector(labels, targets_test)
	print("6 - Create Ticket Feature Sequences")
	#Create Sequences Features
	tickets_feature_sequences = dataL.extractFeatures(tickets_training_tl, dictionary)
	tickets_feature_test_sequences = dataL.createDataSequence(tickets_test_tl, dictionary)

	print("6 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_training_sequences, oneHotVectorTarget_training, tickets_feature_sequences_training, trash = filtdata.removeTokenOOVwithSequenceFeatures(tickets_training_sequences, oneHotVectorTarget_training, tickets_feature_sequences, dictionary)
	print("*** Classe Cestino in Training : " + str(len(trash)))
	tickets_test_sequences, oneHotVectorTarget_test, tickets_feature_test_sequences, trash = filtdata.removeTokenOOVwithSequenceFeatures(tickets_test_sequences,oneHotVectorTarget_test, tickets_feature_test_sequences, dictionary)
	print("*** Classe Cestino in Test : " + str(len(trash)))
	print("7 - Generate Training and Testing Dataset\n")
	dataL.writeArrayStringInFile(tickets_training_sequences, 'parsed_sequences/tickets_training.txt', "utf-8")
	dataL.writeArrayStringInFile(tickets_test_sequences, 'parsed_sequences/tickets_test.txt', "utf-8")
	dataL.writeArrayStringInFile(oneHotVectorTarget_training, 'parsed_sequences/target_training.txt', "utf-8")
	dataL.writeArrayStringInFile(oneHotVectorTarget_test, 'parsed_sequences/target_test.txt', "utf-8")
	print("*** Training Size : " + str(len(tickets_training_sequences)) + "\n")
	if config.use_pretrained_embs:
		print("*** Uso pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel,reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	print("8 - Start Training\n")
	ml.runTrainingWithFeatureSequence(config, tickets_training_sequences, oneHotVectorTarget_training, labels, tickets_feature_sequences_training)
	print("============ End =============\n")

"""##################################################################"""

def mainTrainModelOnPreprocessedData():
	print("============ Start =============\n")
	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets Sequences\n")
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	tickets = dataL.loadDataInArray(config.main_path + "parsed_sequences_b/tickets_training.txt", config.csv_encoding)
	targets = dataL.loadDataInArray(config.main_path + "parsed_sequences_b/target_training.txt")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	oneHotVectorTarget = dataL.transformListStringInOneHot(targets)
	print("*** Training Size : " + str(len(tickets)) + "\n")
	tickets_parsed = []
	for t in tickets:
		tickets_work = []
		tt = re.split("\[", t)
		tt = re.split("\]", tt[1])
		tt = re.split(",", tt[0])
		for inner_t in tt:
			a = int(inner_t)
			tickets_work.append(a)
		tickets_parsed.append(tickets_work)

	print("3 - Load Vocabulary\n")
	voc = vc.Vocabulary(config)
	dictionary = voc.loadDictionary("vocabulary")
	reverse_dict = voc.getReverseDictionary(dictionary)

	print("*** Training Size : " + str(len(tickets)) + "\n")
	if config.use_pretrained_embs:
		print("*** Uso pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	print("4 - Start Training\n")
	ml.runTraining(config, tickets_parsed, oneHotVectorTarget, labels)
	print("============ End =============\n")

"""##################################################################"""

def mainEvalModel():
	print("============ Start =============\n")
	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets Sequences\n")
	map_labels = dataL.loadMapFromJson(config.data_path + "map_labels.json")
	tickets = dataL.loadDataInArray(config.main_path + "parsed_sequences/tickets_test.txt", config.csv_encoding)
	targets = dataL.loadDataInArray(config.main_path + "parsed_sequences/target_test.txt")
	labels = dataL.getfirstLevelTargets(map_labels['map'])
	oneHotVectorTarget = dataL.transformListStringInOneHot(targets)
	print("*** Test Size : " + str(len(tickets)) + "\n")
	tickets_parsed = dataL.fromSequenceStringToSequenceArray(tickets)

	print("3 - Load Vocabulary\n")
	voc = vc.Vocabulary(config)
	dictionary = voc.loadDictionary("vocabulary")
	# Create Sequence
	#tickets_sequences = dataL.createDataSequence(tickets_to_lower, dictionary)
	# Create OneHotVector for the targets
	#oneHotVectorTarget = dataL.transformInOneHotVector(labels, targets)
	print("4 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_training_sequences, oneHotVectorTarget_training, trash = filtdata.removeTokenOOV(tickets_parsed,oneHotVectorTarget,dictionary)
	print("*** Class Trash len : " + str(len(trash)))

	print("5 - Restore and Eval Model - " + str(config.model_to_restore))
	#
	#tm.restoreModel(config,tickets_parsed,oneHotVectorTarget, labels, dictionary)
	#tm.restoreModelAndPredict(config,tickets_parsed,oneHotVectorTarget, labels, dictionary)
	ml.runEvaluation(config, tickets_parsed, oneHotVectorTarget, labels, dictionary)
	#
	print("============ End =============\n")


#with tf.device("/gpu:0"):
def main():
	config = cg.Config()
	initialization(config)

	if config.is_test :
		print("======================***** Eval On Trained Model *****======================")
		mainEvalModel()
	else:
		print("======================***** Training Model *****======================")
		#mainTrainModelOnApertureWithSequenceFeatures()
		mainTrainModelOnAperture()
		#mainTrainModel()
		#mainTrainModelOnPreprocessedData()

#main()


def getPriorityTarget():
	configModelPriority = cg.Config()
	configModelPriority.configFromFile("config/priority_config.json")

	dataL = dt.Data(configModelPriority)
	tickets = dataL.loadDataInArray(configModelPriority.main_path + "onlyAperturaPriority/tickets_test.txt", configModelPriority.csv_encoding)
	targets = dataL.generatePriorityTarget(tickets)
	dataL.writeArrayInFileCompleteDataPath(targets,configModelPriority.main_path + "/onlyAperturaPriority/targets_test.txt", "utf-8")

#getPriorityTarget()


def trainPriority():
	print("============ Start =============\n")

	print("1 - Load Configuration\n")
	config = cg.Config()
	config.configFromFile("config/priority_config.json")
	dataL = dt.Data(config)
	print("2 - Load Data and Targets\n")
	tickets_training = dataL.loadDataInArray(config.main_path + "onlyAperturaPriority/tickets_training.txt",config.csv_encoding)
	tickets_test = dataL.loadDataInArray(config.main_path + "onlyAperturaPriority/tickets_test.txt",config.csv_encoding)
	targets_training = dataL.loadDataInArray(config.main_path + "onlyAperturaPriority/targets_training.txt",config.csv_encoding)
	targets_test = dataL.loadDataInArray(config.main_path + "onlyAperturaPriority/targets_test.txt",config.csv_encoding)
	labels = ["1", "2", "3", "4", "5"]
	print("3 - Preprocess Data\n")
	tickets_training_tl, targets_training, words = preprocessData(tickets_training, targets_training, labels)
	tickets_test_tl, targets_test, w_ = preprocessData(tickets_test, targets_test, labels)
	if config.loadOrbuild_dictionary == "build":
		print("4 - Build Vocabulary\n")
		# Create Vocabulary
		voc = vc.Vocabulary(config)
		dictionary, reverse_dict = voc.build_dictionary(words, labels)
		voc.saveDictionary(dictionary, "vocabulary")
		print("*** Vocabulary saved \n")
	else:
		print("4 - Load Vocabulary\n")
		# Load Existing Vocabulary
		voc = vc.Vocabulary(config)
		dictionary = voc.loadDictionary("vocabulary")
		reverse_dict = voc.getReverseDictionary(dictionary)

	print("5 - Create Ticket Sequences and Targets Hot Vectors\n")
	# Create Sequences and HotVectors for the Target
	tickets_training_sequences = dataL.createDataSequence(tickets_training_tl, dictionary)
	oneHotVectorTarget_training = dataL.transformInOneHotVector(labels, targets_training)
	tickets_test_sequences = dataL.createDataSequence(tickets_test_tl, dictionary)
	oneHotVectorTarget_test = dataL.transformInOneHotVector(labels, targets_test)
	print("6 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_training_sequences, oneHotVectorTarget_training, trash = filtdata.removeTokenOOV(tickets_training_sequences,
																							 oneHotVectorTarget_training,
																							 dictionary)
	print("	*** Classe Cestino in Training : " + str(len(trash)) + "\n")
	tickets_test_sequences, oneHotVectorTarget_test, trash = filtdata.removeTokenOOV(tickets_test_sequences,
																					 oneHotVectorTarget_test,
																					 dictionary)
	print("	*** Classe Cestino in Test : " + str(len(trash)) + "\n")
	print("7 - Generate Training and Testing Dataset\n")
	dataL.writeArrayInFileCompleteDataPath(tickets_training_sequences,config.data_sequences_path + '/tickets_training.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(tickets_test_sequences, config.data_sequences_path + '/tickets_test.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(oneHotVectorTarget_training, config.data_sequences_path + '/target_training.txt', "utf-8")
	dataL.writeArrayInFileCompleteDataPath(oneHotVectorTarget_test, config.data_sequences_path + '/target_test.txt', "utf-8")
	print("	*** Training Size : " + str(len(tickets_training_sequences)) + "\n")
	print("	*** Test Size : " + str(len(tickets_test_sequences)) + "\n")
	if config.use_pretrained_embs:
		print("	*** Use pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	print("8 - Start Training\n")
	ml.runTraining(config, tickets_training_sequences, oneHotVectorTarget_training, labels)
	print("============ End =============\n")


trainPriority()

def testPriority():
	print("============ Start =============\n")
	print("1 - Load Configuration\n")
	config = cg.Config()
	dataL = dt.Data(config)
	print("2 - Load Data and Targets Sequences\n")

	tickets = dataL.loadDataInArray(config.main_path + "parsed_sequences/tickets_test.txt", config.csv_encoding)
	targets = dataL.loadDataInArray(config.main_path + "parsed_sequences/target_test.txt")
	labels = ["1", "2" , "3", "4", "5"]
	oneHotVectorTarget = dataL.transformListStringInOneHot(targets)
	print("*** Test Size : " + str(len(tickets)) + "\n")
	tickets_parsed = dataL.fromSequenceStringToSequenceArray(tickets)

	print("3 - Load Vocabulary\n")
	voc = vc.Vocabulary(config)
	dictionary = voc.loadDictionary("vocabulary")
	# Create Sequence
	# tickets_sequences = dataL.createDataSequence(tickets_to_lower, dictionary)
	# Create OneHotVector for the targets
	# oneHotVectorTarget = dataL.transformInOneHotVector(labels, targets)
	print("4 - Filter Data - Removeing Token OOV\n")
	filtdata = fd.FilterData(config, labels)
	tickets_training_sequences, oneHotVectorTarget_training, trash = filtdata.removeTokenOOV(tickets_parsed,oneHotVectorTarget,dictionary)
	print("*** Class Trash len : " + str(len(trash)))

	print("5 - Restore and Eval Model - " + str(config.model_to_restore))
	#
	# tm.restoreModel(config,tickets_parsed,oneHotVectorTarget, labels, dictionary)
	# tm.restoreModelAndPredict(config,tickets_parsed,oneHotVectorTarget, labels, dictionary)
	ml.runEvaluation(config, tickets_parsed, oneHotVectorTarget, labels, dictionary)
	#
	print("============ End =============\n")

#testPriority()
