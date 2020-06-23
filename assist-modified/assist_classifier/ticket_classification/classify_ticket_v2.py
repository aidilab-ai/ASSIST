import config_in as cg
import connector as con
import model as ml
import tokenizer as tk
import vocabulary as vc
import data as dt
import filter_data as fd
import skipgram as sk
import logger as lg
import requests
import tensorflow as tf
import sys
import json


#
QIlogger = lg.getLogger("__main__")

def cleanData(data):
	tok = tk.Tokenizer(data)
	tok.tokenizeTickets()
	tickets_to_lower = tok.toLower()
	tickets_no_sw = tok.removeStopWordsFromTicket(tickets_to_lower[0])
	return tickets_no_sw

def classifyNextTicket(config_path, customer):
	#lg.configureLogger(QIlogger, customer, "classify")
	QIlogger.info("----------------------------------------------------------------------")
	QIlogger.info("==================Ticket Classification Start=========================")

	QIlogger.info("** Initialization Start**")
	config_directory = config_path  + customer + "/"
	config_category_file = "category_config.json"
	config_priority_file = "priority_config.json"
	config_connector_file = "connector_config.json"
	#
	QIlogger.info("1 - Load Configurations start...")

	QIlogger.info("	** Config Path " + config_directory)
	try:
		#Load Config files
		QIlogger.info("	** Config for Class Classification")
		configModelClass = cg.Config()
		print("	** Config Path " + config_directory)
		configModelClass.configFromFile(config_directory + config_category_file)
		configModelClass.is_test = True
		labelsClass = configModelClass.labels
		dataMC = dt.Data(configModelClass)
		#
		QIlogger.info("	** Config for Priority Classification")
		configModelPriority = cg.Config()
		configModelPriority.configFromFile(config_directory + config_priority_file)
		configModelPriority.is_test = True
		labelsPriority = configModelPriority.labels
		#dataMC = dt.Data(configModelPriority)

		QIlogger.info(" ** Load Vocabulary")
		# Load Existing Vocabulary
		voc = vc.Vocabulary(configModelClass)
		dictionary = voc.loadDictionary("vocabulary")
		reverse_dict = voc.getReverseDictionary(dictionary)

		if configModelClass.use_pretrained_embs:
			QIlogger.info("	** Uso pretrained Words Embedding\n")
			skip = sk.SkipgramModel(configModelClass)
			skipgramModel = skip.get_skipgram()
			voc = vc.Vocabulary(configModelClass)
			reverse_dict = voc.getReverseDictionary(dictionary)
			skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
			configModelClass.skipgramEmbedding = skipgramEmbedding

			voc_priority = vc.Vocabulary(configModelPriority)
			dictionary_priority = voc_priority.loadDictionary("vocabulary")
			reverse_dict_priority = voc.getReverseDictionary(dictionary_priority)
			skipgramEmbedding_p = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict_priority)
			configModelPriority.skipgramEmbedding = skipgramEmbedding_p

		QIlogger.info("1 - Load Configuration end")
		#

		QIlogger.info("2 - Login In API")
		# Login to API
		configConnection = con.ConfigConnection()
		configConnection.configFromFile(config_directory + config_connector_file)
		connector = con.Connector(configConnection)
		# Create Persistent Session
		Reqsess = requests.session()
		# LogIN
		connector.login(Reqsess)
		next_exist = True
		numb_get = 0
		webService = configConnection.getWebService()
		num_nex_tickets = webService['numb_next_tickets']

		ids, descriptions, sequences, categoryIDS, priorityIDS = [], [], [], [], []
		while next_exist and numb_get <= num_nex_tickets:
			ticket = None
			ticket_id = None
			QIlogger.info("3 - Get Next Tickets")
			# chiamo next ticket
			response = connector.getNextTicket(Reqsess)
			# Check if response is empty or not
			if not response:
				next_exist = False
			else:
				ticket_id  = str(response['id'])

			if response['id'] != 0:
				numb_get = numb_get + 1
				ticket = [str(response['description'])]
				QIlogger.info("	** Ticket : " + str(ticket))
				#filtraggio e creazione della sequenza
				QIlogger.info("4 - Clean Ticket")
				ticket_cleaned = cleanData(ticket)
				ticket_array = ticket_cleaned.split(" ")
				QIlogger.info(" ** Clean data " + str(ticket_array))
				QIlogger.info("5 - Ticket Sequence Creation")
				# Create Sequences and HotVectors for the Target
				tickets_sequences = dataMC.createDataSequence([ticket_array], dictionary)
				QIlogger.info(" ** Ticket sequences " +str(tickets_sequences))
				ticket_sequence = tickets_sequences[0]
				#
				ids.append(ticket_id)
				descriptions.append(ticket)
				sequences.append(ticket_sequence)
				#
				QIlogger.info("6 - Filter Data")
				#Trashing ticket with too much words out of vocabulary
				filtdata = fd.FilterData(configModelClass, labelsClass)
				trashIT = filtdata.trashingTicket(ticket_sequence, dictionary)
				if trashIT :
					QIlogger.info("7 - Ticket Classified as Trash")
					params = {"categoryId":"Trash", "priorityId":0}
					#invio il ticket con la label : Cestino
					categoryIDS.append("Trash")
					priorityIDS.append(1)
				else :
					categoryIDS.append(0)
					priorityIDS.append(1)
			else:
				next_exist = False

		#Aggrego i tickets
		tickets = [{"id": i, "description": d, "sequence" : s, "categoryID" : c, "priorityID" : p} for i, d, s, c, p in zip(ids, descriptions, sequences, categoryIDS, priorityIDS)]
		#
		#Carico il Modello delle categorie

		# Aggiorno l'oggetto
		if len(tickets) > 0:
			QIlogger.info("8 - Run Category Classification")

			QIlogger.info("	** Category Prediction")
			out_array, class_predicted = ml.runPredictionTickets(configModelClass, tickets, labelsClass)

			for i in range(len(tickets)):
				if tickets[i]['categoryID'] != "Trash":
					tickets[i]['categoryID'] = configModelClass.labels_map.get(class_predicted[i])
					QIlogger.info("	** Categories Predicted : " + str(class_predicted))
				else:
					tickets[i]['categoryID'] = 0
					QIlogger.info("	** Categories Predicted TRASH")
			# Carico il Modello delle priorita
			QIlogger.info("9 - Run Priority Classification")
			QIlogger.info("	** Priority Prediction\n")
			out_array, class_predicted = ml.runPredictionTickets(configModelPriority, tickets, labelsPriority)
			# Aggiorno l'oggetto
			for i in range(len(tickets)):
				if tickets[i]['categoryID'] != 0:
					tickets[i]['priorityID'] = labelsPriority.index(class_predicted[i]) + 1

			#Invio la classificazione
			for ticket in tickets:
				ticketID = ticket['id']
				categoryID = ticket['categoryID']
				priorityID = ticket['priorityID']
				params = {"categoryId": categoryID, "priorityId": priorityID}
				QIlogger.info("10 - Send Classification")
				connector.sendClassification(Reqsess, ticketID, params)
		else:
			QIlogger.info("Non ci sono ticket da classificare. Chiudo il processo.")

		# LogIN
		QIlogger.info("LogOut dalle API")
		connector.logout(Reqsess)

	except Exception as e:
		print(str(e))
		QIlogger.error("Error in classifyNextTicket " + str(e))
	#LogOut
	#connector.logout(Reqsess)
	QIlogger.info("==================Ticket Classification End=========================\n")
	QIlogger.info("----------------------------------------------------------------------")


#if __name__ == "__main__":
	#
	#lg.configureLogger(QIlogger, "antonio", "training")
	#classifyNextTicket("C:/Users/Antonio/git/seq2seq_tensorflow/src/nn/ticket_classification/config/", "antonio")
	#classifyNextTicket(config_path=sys.argv[1])
