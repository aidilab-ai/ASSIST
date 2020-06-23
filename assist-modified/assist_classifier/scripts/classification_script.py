import os
import sys
import logger as lg
import demon as dm
import config as cg

import sys
# Add the ptdraft folder path to the sys.path list
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path +'/../ticket_classification/')
# Now you can import your module
import classify_ticket_v2 as ct
import logger as clLog
import time
from random import randint
#
QIlogger = lg.getLogger(__name__)


"""##########################################################################"""
"""CLASSIFICATION Class"""
"""##########################################################################"""
class Classification(object):
	def __init__(self, user):
		#
		self.home_path = ""
		self.code_path = ""
		self.log_file = ""
		#
		self.name = user["name"]
		self.config_directory = user["config_dir"]

	"""##########################################################################"""

	def getHomePath(self):
		return self.home_path

	def getCodePath(self):
		return self.code_path

	def logFile(self):
		return self.log_file

	def getName(self):
		return self.name

	def getConfigDirectory(self):
		return self.config_directory


	"""##########################################################################"""

"""
####### CLASSIFICATION SCRIPT that run on all users #######
"""
def classification():
	classificationDemon = dm.Demon("classification.pid")
	trainingDemon = dm.Demon("training.pid")
	try:
		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("1 - Classification Script")

		QIlogger.info("2 - Check Demon Existence")
		#
		while(classificationDemon.checkPidDemon()):
			ctime = classificationDemon.getCreationTime()
			today = classificationDemon.getCurrentTimeStamp()
			days = classificationDemon.findDaysBetweenTimeStamps(today, ctime)
			if days < 1:
				r = randint(1, 30)
				QIlogger.info("-- Pid Exist. Wait " + str(r) + " sec")
				time.sleep(r)
			else:
				QIlogger.info(
					"-- The past demon has been created " + str(days) + " days ago. Delete it to prevent stall")
				classificationDemon.deletePidDemon()
		#
		QIlogger.info("3 - Create Classification PID file")
		classificationDemon.createPidDemon()

		#
		QIlogger.info("4 - Load Configuration")
		config = cg.Config()
		dir_path = os.path.dirname(os.path.realpath(__file__))
		config.configFromFile(dir_path + "/config/config.json")
		#
		users = config.users
		#
		for user in users:
			QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++")
			QIlogger.info("5 - User Selected : " + str(user["name"]))
			classification = Classification(user)

			#clLog.configureLogger(QIlogger, user["name"], "classification")
			dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../ticket_classification/config/"
			QIlogger.info("6 - Open Classification Script For user : " + str(user["name"]))
			ct.classifyNextTicket(dir_path,user["name"])
			QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++\n")
		#
		QIlogger.info("7 - Delete PID File")
		classificationDemon.deletePidDemon()
		QIlogger.info("------------------------------------------------------\n")
		#
	except Exception as e:
		classificationDemon.deletePidDemon()
		QIlogger.error("Error in classification method " + str(e))

	"""##########################################################################"""

"""
####### CLASSIFICATION SCRIPT that run on all users #######
"""
def classification_user(user_name):
	classificationDemon = dm.Demon("classification_" + user_name + ".pid")
	trainingDemon = dm.Demon("training.pid")

	try:
		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("1 - Classification Script")

		QIlogger.info("2 - Check Demon Existence")
		# Check dell'esistenza di un pid file della classificazione. Nel caso esista usciamo
		if (classificationDemon.checkPidDemon()):
			ctime = classificationDemon.getCreationTime()
			today = classificationDemon.getCurrentTimeStamp()
			days = classificationDemon.findDaysBetweenTimeStamps(today, ctime)
			if days < 1:
				QIlogger.info("-- Pid Exist. EXIT.")
				return False
			else:
				QIlogger.info("-- The past demon has been created " + str(days) + " days ago. Delete it to prevent stall")
				classificationDemon.deletePidDemon()

		#
		QIlogger.info("3 - Create Classification PID file")
		classificationDemon.createPidDemon()

		#
		QIlogger.info("4 - Load Configuration")
		config = cg.Config()
		dir_path = os.path.dirname(os.path.realpath(__file__))
		config.configFromFile(dir_path + "/config/config.json")
		#
		users = config.users
		#
		for user in users:
			if user["name"] == user_name:
				QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++")
				QIlogger.info("5 - User Selected : " + str(user["name"]))
				classification = Classification(user)

				#clLog.configureLogger(QIlogger, user["name"], "classification")
				dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../ticket_classification/config/"
				QIlogger.info("6 - Open Classification Script For user : " + str(user["name"]))
				ct.classifyNextTicket(dir_path,user["name"])
				QIlogger.info("++++++++++++++++++++++++++++++++++++++++++++\n")
		#
		QIlogger.info("7 - Delete PID File")
		classificationDemon.deletePidDemon()
		QIlogger.info("------------------------------------------------------\n")
		#
	except Exception as e:
		classificationDemon.deletePidDemon()
		QIlogger.error("Error in classification method " + str(e))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""CLASSIFICATION"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def main():
	lg.configureLogger(QIlogger, "", "classification")
	classification()


def main_user():
	if len(sys.argv) < 1:
		print("add user name to run the script.")
		exit(1)
	#
	user_name = sys.argv[1]
	lg.configureLogger(QIlogger, user_name, "classification")
	classification_user(user_name)

""""""
if __name__ == '__main__':
	#main()
	main_user()
