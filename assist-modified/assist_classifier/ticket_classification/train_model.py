import tensorflow as tf

import config_in as cg
import connector as con
import data as dt
import filter_data as fd
import model as ml
import tokenizer as tk
import utility as ut
import vocabulary as vc
import skipgram as sk
import logger as lg
#import nn.assist_classifier.scripts.logger as lg

import requests
import os

#
QIUserLogger = lg.getLogger("__main__")



def gatherData(type, rawData, config, id2lab):
	#
	tickets = []
	targets = []

	#
	for ticket in rawData:
		tickets.append(ticket['description'])
		if type == 'category':
			id = ticket['category_id']
			targets.append(id2lab.get(id))
		else :
			targets.append(str(ticket['priority']))
	return tickets,targets

def training_model(main_path, type, config_file, from_date, to_date, customer):

	#logging.basicConfig(filename=logCustomer, level=logging.INFO)
	#lg.configureLogger(QIUserLogger, customer, "training")
	#
	QIUserLogger.info("-----------------------------------------------------------------")
	QIUserLogger.info("------------------------Training Start---------------------------")
	#
	QIUserLogger.info("** Initialization start... **")
	main_path = main_path
	type = type
	config_file = config_file
	from_date = from_date
	to_date = to_date

	QIUserLogger.info("	MainPath - " + str(main_path))
	QIUserLogger.info("	Type - " + str(type))
	QIUserLogger.info("	ConfigFile - " + str(config_file))
	QIUserLogger.info("	FromDate - " + str(from_date))
	QIUserLogger.info("	ToDate - " + str(to_date))
	#
	QIUserLogger.info("** Initialization End **")

	try:
		QIUserLogger.info("1 - Load Configurations")
		QIUserLogger.info("	** Config for Classification")
		# Load Config files
		configModel = cg.Config()
		configModel.configFromFile(config_file)
		configModel.main_path = main_path
		configModel.updateDataOfMainPath(config_file, main_path)
		dataL = dt.Data(configModel)
		#
		QIUserLogger.info("2 - Login In API")
		# Login to API
		configConnection = con.ConfigConnection()

		dir_path = os.path.dirname(os.path.realpath(__file__))
		configConnection.configFromFile(dir_path + "/config/" + customer + "/connector_config.json")
		connector = con.Connector(configConnection)
		# Create Persistent Session
		Reqsess = requests.session()
		# LogIN
		connector.login(Reqsess)
		QIUserLogger.info("3 - GET TICKETS FROM API")
		#
		params = "closedfrom=" + str(from_date) + "&closedto=" + str(to_date) + "&maxnum=" + str(configConnection.max_tickets_to_get)
		#params = {"closedfrom": from_date, "closedto": to_date, "maxnum" : configConnection.max_tickets_to_get}
		responseTicket = connector.getTickets(Reqsess,params)
		if len(responseTicket) > 0 :
			rTicket = []
			for t in responseTicket:
				rTicket.append(t['description'])
			#
			id2lab = dict(zip(configModel.labels_map.values(), configModel.labels_map.keys()))
			#
			gather_tickets, gather_targets = gatherData(type, responseTicket, configModel, id2lab)
			#
			QIUserLogger.info("4 - REMOVE STOP WORDS FROM NEW TICKETS")
			tok = tk.Tokenizer(gather_tickets)
			tok.tokenizeTickets()
			tickets_to_lower = tok.toLower()
			gather_tickets, gather_targets = tok.removeStopWordsToString(tickets_to_lower, gather_targets)

			QIUserLogger.info("5 - GET STORED DATA TICKETS")
			tickets_train = dataL.loadDataInArray(configModel.data_path + "/tickets.txt", configModel.csv_encoding)
			targets_train = dataL.loadDataInArray(configModel.data_path + "/targets.txt")
			#
			# Count if we reached the threshold
			QIUserLogger.info("6 - MERGE THE DATA - STORED AND GATHERED")
			max_length = configModel.max_num_tickets
			len_gather_tickets = len(gather_tickets)
			len_tickets = len(tickets_train)
			#Effettuo un nuovo training su tutto il dataset e non un transfer
			#learning perchè voglio utilizzare sempre un vocabolario aggiornato.
			tickets = tickets_train + gather_tickets
			targets = targets_train + gather_targets
			reached_dim = len_gather_tickets + len_tickets
			if reached_dim > max_length:
				elem_to_cut = reached_dim - max_length
				#cut out the firsts elem_to_cut elements
				merged_targets = tickets[elem_to_cut:]
				merged_tickets = targets[elem_to_cut:]
				tickets = merged_tickets
				targets = merged_targets
				reached_dim = max_length


			QIUserLogger.info("7 - REMOVE IDENTICAL TICKETS")
			#tickets, targets = ut.removeIdenticalTickets(tickets, targets)
			tickets, targets = ut.removeIdenticalTicketsFromNew(tickets, targets,len_tickets,reached_dim)

			QIUserLogger.info("8 - SAVING MERGED DATA")
			dataL.writeArrayInFileCompleteDataPath(tickets, configModel.data_path + '/tickets.txt', "utf-8")
			dataL.writeArrayInFileCompleteDataPath(targets, configModel.data_path + '/targets.txt', "utf-8")
			#
			QIUserLogger.info("9 - EXTRACT WORDS FROM TICKETS")
			words = tok.extractWordsTicketString(tickets)
			#
			QIUserLogger.info("10 - BUILD NEW VOCABULARY")
			# Create Vocabulary
			voc = vc.Vocabulary(configModel)
			dictionary, reverse_dict = voc.build_dictionary(words, configModel.labels)
			voc.saveDictionary(dictionary, "vocabulary")
			QIUserLogger.info("*** Vocabulary saved")
			#
			QIUserLogger.info("11 -- SPLIT DATA IN TRAINING AND TEST DATASET")
			tickets_training, tickets_test, Target_training, Target_test = ut.get_train_and_test(tickets, targets)
			dataL.writeArrayInFileCompleteDataPath(tickets_training, configModel.data_path + '/tickets_training.txt', "utf-8")
			dataL.writeArrayInFileCompleteDataPath(Target_training, configModel.data_path + '/targets_training.txt', "utf-8")

			dataL.writeArrayInFileCompleteDataPath(tickets_test, configModel.data_path + '/tickets_test.txt', "utf-8")
			dataL.writeArrayInFileCompleteDataPath(Target_test, configModel.data_path + '/targets_test.txt', "utf-8")


			#
			QIUserLogger.info("12 - CREATE TICKETS AND TARGETS SEQUENCES")
			# Create Sequences and HotVectors for the Target
			tickets_training_sequences = dataL.createDataSequenceTicketsString(tickets_training, dictionary)
			oneHotVectorTarget_training = dataL.transformInOneHotVector(configModel.labels, Target_training)
			#
			QIUserLogger.info("13 - FILTER OUT  DATA - Removing Token OOV")
			filtdata = fd.FilterData(configModel, configModel.labels)
			tickets_training_sequences, oneHotVectorTarget_training, trash = filtdata.removeTokenOOV(tickets_training_sequences,oneHotVectorTarget_training,dictionary)
			QIUserLogger.info("	*** Classe Cestino in Training : " + str(len(trash)))
			#
			#QIUserLogger.info("	-- Split Training | Test Dataset")
			#tickets_training_sequences, tickets_test_sequences, oneHotVectorTarget_training, oneHotVectorTarget_test = ut.get_train_and_test(tickets_training_sequences, oneHotVectorTarget_training)
			#
			QIUserLogger.info("14 - SAVING TRAINING SEQUENCES")
			dataL.writeArrayInFileCompleteDataPath(tickets_training_sequences,configModel.data_sequences_path + '/tickets_training.txt', "utf-8")
			dataL.writeArrayInFileCompleteDataPath(oneHotVectorTarget_training,configModel.data_sequences_path + '/target_training.txt', "utf-8")

			QIUserLogger.info("	*** Training Size : " + str(len(tickets_training_sequences)) + "\n")
			if configModel.use_pretrained_embs:
				QIUserLogger.info("	*** Use pretrained Words Embedding")
				skip = sk.SkipgramModel(configModel)
				skipgramModel = skip.get_skipgram()
				skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
				configModel.skipgramEmbedding = skipgramEmbedding
				# Start Training
				QIUserLogger.info("15 - START TRAINING")
			ml.runTraining(configModel, tickets_training_sequences, oneHotVectorTarget_training, configModel.labels)
			QIUserLogger.info("============ End =============")
		else:
			QIUserLogger.info("No New Tickets found. There is no need of a new training.")

		# LogIN
		connector.logout(Reqsess)
		#
	except Exception as e:
		print(str(e))
		QIUserLogger.error("Error in training_model " + str(e))

#if __name__ == "__main__":
	#customer = sys.argv[1]
	#main_path = sys.argv[2]
	#type = sys.argv[3]
	#config_file = sys.argv[4]
	#from_date = sys.argv[5]
	#to_date = sys.argv[6]
	#training_model("C:/Users/Antonio/git/seq2seq_tensorflow/src/nn/ticket_classification/config/", "antonio")
	#customer = "antonio"
	#main_path = "C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/priority/model/data"
	#type = "priority"
	#config_file = "C:/Users/Antonio/git/seq2seq_tensorflow/src/nn/assist_classifier/ticket_classification/config/antonio/" + type + "_config.json"
	#from_date = "20051231"
	#to_date = "20190218"
	#
	#lg.configureLogger(QIUserLogger, customer, "training")
	#training_model(main_path, type, config_file, from_date, to_date, customer)
