import os
import sys


import logger as lg
import demon as dm
import config as cg
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path +'/../ticket_classification/')


import train_model as tm
import logger as clLog
import timeLogger as tl
import evalute_models as ev
import time
import datetime
import shutil
from random import randint
import re
#
QIlogger = lg.getLogger(__name__)
#QIUserLogger = lg.getLogger("user")

"""##########################################################################"""
"""TRAINING Class"""
"""##########################################################################"""
class Training(object):

	def __init__(self, user, config):
		#
		self.home_path = config.home_path
		#
		self.name = user["name"]
		self.config_directory = user["config_dir"]
		self.model_type = ""
		self.log_file = ""
		self.date_log_file = ""
		self.new_model_dir = ""
		self.model_dir = ""
		self.old_model_dir = ""
		self.data_dir = "data"
		self.saved_models_dir = "saved_model"
		self.best_models_dir = "best_models"
		self.tensorboard_dir = "tensorboard"
		self.parsed_sequences_dir = "parsed_sequences"
		self.current_date = datetime.datetime.today().strftime('%Y-%m-%d')
		self.last_date = ""

	"""##########################################################################"""

	def setModelType(self, type):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_type = type
		self.date_log_file = "last_date.json"
		self.new_model_dir = self.home_path + "/" + self.name + "/models/" + self.model_type + "/model_new"
		self.model_dir = self.home_path + "/" + self.name + "/models/" + self.model_type + "/model"
		self.old_model_dir = self.home_path + "/" + self.name + "/models/" + self.model_type + "/model_old"


	"""##########################################################################"""

	def getModelType(self):
		return self.model_type

	def setLastDate(self, last_date):
		self.last_date = last_date

	def getLastDate(self):
		return self.last_date

	def getName(self):
		return self.name

	def getConfigDirectory(self):
		return self.config_directory

	def getDateLogFile(self):
		return self.date_log_file

	def getNewModelDir(self):
		return self.new_model_dir

	def getModelDir(self):
		return self.model_dir

	def getOldModelDir(self):
		return self.old_model_dir

	def getSavedModelDir(self):
		return self.saved_models_dir

	def getBestModelDir(self):
		return self.best_models_dir

	def getTensorBoardDir(self):
		return self.tensorboard_dir

	def getParsedSequencesDir(self):
		return self.parsed_sequences_dir

	def getCurrentDate(self):
		return self.current_date

	def getDataDir(self):
		return self.data_dir


	"""##########################################################################"""

	def createNewModelDirs(self):
		try:
			if not os.path.exists(self.new_model_dir):
				os.makedirs(self.new_model_dir,0o755)
			if not os.path.exists(self.new_model_dir + "/" + self.saved_models_dir):
				os.makedirs(self.new_model_dir + "/" + self.saved_models_dir,0o775)
			if not os.path.exists(self.new_model_dir + "/" + self.best_models_dir):
				os.makedirs(self.new_model_dir + "/" + self.best_models_dir,0o775)
			if not os.path.exists(self.new_model_dir + "/" + self.tensorboard_dir):
				os.makedirs(self.new_model_dir + "/" + self.tensorboard_dir,0o775)
			if not os.path.exists(self.new_model_dir + "/" + self.parsed_sequences_dir):
				os.makedirs(self.new_model_dir + "/" + self.parsed_sequences_dir,0o775)
			if not os.path.exists(self.new_model_dir + "/" + self.data_dir):
				os.makedirs(self.new_model_dir + "/" + self.data_dir,0o775)
		except OSError:
			print("Creation of the directory failed")
			QIlogger.error("Creation of the directories failed")


	"""##########################################################################"""

	def copyDataFromModelToNewModel(self):
		try:
			src_files = os.listdir(self.model_dir + "/" + self.data_dir)
			for file_name in src_files:
				full_file_name = os.path.join(self.model_dir + "/" + self.data_dir, file_name)
				if (os.path.isfile(full_file_name)):
					shutil.copy(full_file_name, self.new_model_dir + "/" + self.data_dir)
		except Exception as e:
			QIlogger.error("Impossible to copy files to new models "+ str(e))


	"""##########################################################################"""

	def removeDir(self, dir):
		try:
			if os.path.exists(dir):
				shutil.rmtree(dir, ignore_errors=True)
		except Exception as e:
			QIlogger.error("Impossible to delete old model directory " + str(e))


	"""##########################################################################"""

	def moveFileToDir(self, sourceDir, destDir):
		try:
			source = os.listdir(sourceDir)
			if not os.path.exists(destDir):
				os.makedirs(destDir, 0o755)
			for files in source:
				shutil.move(sourceDir+"/"+files, destDir)
		except Exception as e:
			QIlogger.error("Impossible to move files to directory " + str(e))


	"""##########################################################################"""

	def moveSpecificFilesToDir(self, sourceDir, destDir, files):
		try:
			if not os.path.exists(destDir):
				os.makedirs(destDir, 0o755)
			for file in files:
				shutil.move(sourceDir+"/"+file, destDir)
		except Exception as e:
			QIlogger.error("Impossible to move files to directory " + str(e))


	"""##########################################################################"""

	def replaceDataInCheckpoint(self):
		try:
			checkpointPath = self.getModelDir() + "/" + self.getBestModelDir() + "/checkpoint"
			with open(checkpointPath, 'r') as myfile:
				data = myfile.read()

			output = re.sub(self.getNewModelDir(),  self.getModelDir(), data)

			with open(checkpointPath, 'w', encoding="UTF-8") as f:
				f.write("%s\n" % output)

		except Exception as e:
			QIlogger.error("Impossible to replace Data in checkpoint " + str(e))


"""##########################################################################"""
"""Training Script"""
"""##########################################################################"""
def training():
	classificationDemon = dm.Demon("classification.pid")
	trainingDemon = dm.Demon("training.pid")
	try:
		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("1 - Training Script")
		QIlogger.info("2 - Check Demon Existence")
		#
		while (trainingDemon.checkPidDemon()):
			ctime = trainingDemon.getCreationTime()
			today = trainingDemon.getCurrentTimeStamp()
			days = trainingDemon.findDaysBetweenTimeStamps(today, ctime)
			if days < 3 :
				r = randint(1, 30)
				QIlogger.info("-- Pid Exist. Wait " + str(r) + " sec")
				time.sleep(r)
			else :
				QIlogger.info("-- The past demon has been created " + str(days) + " days ago. Delete it to prevent stall")
				trainingDemon.deletePidDemon()
		#
		#
		QIlogger.info("3 - Create Training PID file")
		trainingDemon.createPidDemon()

		#
		QIlogger.info("4 - Load Configuration")
		config = cg.Config()

		dir_path = os.path.dirname(os.path.realpath(__file__))
		config.configFromFile(dir_path +"/config/config.json")
		#
		users = config.users
		#
		lastDateClass = tl.TimeLogger("last_date.json")
		for user in users:
			QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++")
			QIlogger.info("5 - User Selected : " + str(user["name"]))
			training = Training(user,config)
			for model_type in config.model_types:
				QIlogger.info("6 - Model Type : " + str(model_type))
				training.setModelType(model_type)
				QIlogger.info("7 - Check last date from the log date file")

				lastDate = ""
				if lastDateClass.checkFileExistence():
					QIlogger.info("	-- File exists. Get data from there")
					lastDate = lastDateClass.getLastDate()
				else:
					lastDateClass.createTimeFile()
					QIlogger.info("	-- File does not exist. It will be created")
					lastDate = datetime.datetime.today().strftime('%Y-%m-%d')
				#
				training.setLastDate(lastDate)
				QIlogger.info("8 - Create directories for new model")
				training.createNewModelDirs()
				#
				QIlogger.info("9 - Copy data dir from old model to the new one")
				training.copyDataFromModelToNewModel()
				#
				from_date = training.getLastDate()
				from_date_array = from_date.split("-")
				from_date = from_date_array[0] + from_date_array[1] + from_date_array[2];

				to_date = training.getCurrentDate()
				to_date_array = to_date.split("-")
				to_date = to_date_array[0] + to_date_array[1] + to_date_array[2]
				#
				#clLog.configureLogger(QIUserLogger, user["name"], "training")

				dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../ticket_classification/config/"
				QIlogger.info("10 - Open Training Script For user : " + str(user["name"]) + " --------------------- ")
				#ct.classifyNextTicket(dir_path, user["name"])
				try:
					tm.training_model(training.getNewModelDir() + "/", model_type, dir_path+"/"+user["name"]+"/"+model_type+"_config.json", from_date, to_date, user["name"])
				except Exception as e:
					#trainingDemon.deletePidDemon()
					QIlogger.error("Error in training model " + str(e))
				#
				while (classificationDemon.checkPidDemon()):
					r = randint(1, 30)
					QIlogger.info("-- Classification Pid Exist. Wait " + str(r) + " sec")
					time.sleep(r)

				QIlogger.info("11 - Evaluate new model and old model")
				dir_path = os.path.dirname(os.path.realpath(__file__))
				config_path = dir_path + "/../ticket_classification/config/" + user["name"] + "/" + model_type+ "_config.json"
				model_dir = training.getModelDir() + "/"
				new_model_dir = training.getNewModelDir() + "/"
				skipgramModelPath = config.skipgram_model_path
				eval = ev.Evaluator(user, config_path, model_dir, new_model_dir, skipgramModelPath)
				res = eval.runEval()
				if res == 'new_model' :
					QIlogger.info("	-- EVALUATION : New Model has better performance")
					QIlogger.info("	-- Create a Classification DEMON to ensure that classification not start now")
					classificationDemon.createPidDemon()

					QIlogger.info("11 - Remove old model directory")
					training.removeDir(training.old_model_dir)
					#
					QIlogger.info("12 - Move New model as Model and Model as old model")
					os.mkdir(training.getOldModelDir(), 0o775)
					training.moveFileToDir(training.getModelDir(), training.getOldModelDir())
					#delete Model directory
					training.removeDir(training.model_dir)
					#create Model directory
					os.mkdir(training.getModelDir(), 0o775)
					training.moveFileToDir(training.getNewModelDir(), training.getModelDir())
					training.removeDir(training.new_model_dir)

					QIlogger.info("13 - String replace nel file del checkpoint")
					training.replaceDataInCheckpoint()

					QIlogger.info("	-- Destroy classification demon")
					classificationDemon.deletePidDemon()

				else:
					QIlogger.info("	-- EVALUATION : OLD Model has better performance")
					QIlogger.info("11 - Rename tickets and targets in model dir")
					#rename tickets and targets as old_tickets and old_targets in model
					shutil.move(training.getModelDir() + "/data/tickets.txt", training.getModelDir() + "/data/old_tickets.txt")
					shutil.move(training.getModelDir() + "/data/targets.txt",training.getModelDir() + "/data/old_targets.txt")
					#copy the new tickets and targets from new_model
					QIlogger.info("12 - Move tickets and targets from new model to old model")
					training.moveSpecificFilesToDir(training.getNewModelDir() + "/data", training.getModelDir() + "/data", ["tickets.txt", "targets.txt"])
					# delete new Model directory
					QIlogger.info("13 - Remove new model directory")
					training.removeDir(training.getNewModelDir())

				#clLog.removeHandlers(QIUserLogger)
				QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++")

		#
		QIlogger.info("14 - Update date")
		lastDateClass.deleteFile()
		lastDateClass.createTimeFile()

		QIlogger.info("15 - Delete PID File")
		trainingDemon.deletePidDemon()
		QIlogger.info("------------------------------------------------------\n")
	#
	except Exception as e:
		trainingDemon.deletePidDemon()
		QIlogger.error("Error in training method " + str(e))



def main():
	lg.configureLogger(QIlogger, "", "training")
	training()


if __name__ == '__main__':
	main()
