import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path +'/../ticket_classification/')

import config_in as cg
import data as dataL
import vocabulary as vc
import model as ml
import logger as lg


QIlogger = lg.getLogger("__main__")

class Evaluator(object):

	def __init__(self, user, config_path, model_dir, new_model_dir, skipgramModelPath):

		self.name = user["name"]
		self.config_directory = user["config_dir"]
		self.model_dir = model_dir
		self.new_model_dir = new_model_dir
		self.skipGramModelPath = skipgramModelPath
		self.config_path = config_path

	def createConfigModel(self):
		configModel = cg.Config()
		configModel.configFromFile(self.config_path)
		configModel.main_path = self.model_dir
		configModel.updateDataOfMainPath(self.config_path, self.model_dir)
		return configModel

	def createConfigNewModel(self):
		configModel = cg.Config()
		configModel.configFromFile(self.config_path)
		configModel.main_path = self.new_model_dir
		configModel.updateDataOfMainPath(self.config_path, self.new_model_dir)
		return configModel

	def runEval(self):
		QIlogger.info("1 - Create Configuration")
		modelConfig = self.createConfigModel()
		newModelConfig = self.createConfigNewModel()
		model_accuracy = 0.99
		new_model_accuracy = 0.0
		data = dataL.Data(newModelConfig)
		try:
			QIlogger.info("2 - Load Data and Targets")
			tickets = data.loadDataInArray(newModelConfig.main_path + "data/tickets_test.txt", newModelConfig.csv_encoding)
			targets = data.loadDataInArray(newModelConfig.main_path + "data/targets_test.txt")

			QIlogger.info("3 - Load Model Vocabulary")
			voc = vc.Vocabulary(modelConfig)
			dictionary = voc.loadDictionary("vocabulary")

			oneHotVectorTarget = data.transformInOneHotVector(modelConfig.labels, targets)
			tickets_test_sequences = data.createDataSequenceTicketsString(tickets, dictionary)

			model_accuracy = ml.runEvaluationReturnAccuracy(modelConfig, tickets_test_sequences, oneHotVectorTarget, modelConfig.labels, dictionary)
			QIlogger.info("	-- Old Model accuracy : " + str(model_accuracy))
			#

			QIlogger.info("4 - Load New Model Vocabulary")
			voc = vc.Vocabulary(newModelConfig)
			dictionary = voc.loadDictionary("vocabulary")

			oneHotVectorTarget = data.transformInOneHotVector(newModelConfig.labels, targets)
			tickets_test_sequences = data.createDataSequenceTicketsString(tickets, dictionary)

			new_model_accuracy = ml.runEvaluationReturnAccuracy(newModelConfig, tickets_test_sequences, oneHotVectorTarget, newModelConfig.labels, dictionary)
			QIlogger.info("	-- New Model accuracy : " + str(new_model_accuracy))

		except Exception as e:
			QIlogger.error("Error in Evaluate the models " + str(e))

		if(model_accuracy > new_model_accuracy):
			return "old_model"
		else :
			return "new_model"










def main():
	user = {"name" : "antonio" , "config_dir" : ""}
	dir_path = os.path.dirname(os.path.realpath(__file__))
	config_path = dir_path + "/../ticket_classification/config/antonio/category_config.json"
	model_dir = "C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/category/model/"
	new_model_dir = "C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/category/model_new/"
	skipgramModelPath = "C:/Users/Antonio/Desktop/Lavoro/NLG/word_embedding_models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m"
	ticketsPath = "C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/category/model_new/tickets_test.txt"
	targetsPath = "C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/category/model_new/targets_test.txt"
	eval = Evaluator(user, config_path, model_dir, new_model_dir, skipgramModelPath)
	eval.runEval()
	#

#main()
