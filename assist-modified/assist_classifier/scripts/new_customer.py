import os
import sys
import json
import logger as lg
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path +'/../ticket_classification/')
import connector as con
import shutil
import re
from crontab import CronTab
from distutils.dir_util import copy_tree
#
QIlogger = lg.getLogger(__name__)


def createNewDir(dirPath):
	try:
		if not os.path.exists(dirPath):
			os.makedirs(dirPath, 0o755)

	except OSError:
		print("Creation of the directory " + dirPath + " failed")
		QIlogger.error("Creation of the directory "+ dirPath +" failed")

def copyDatasFromToDirs(dirFrom, dirTo):
	try:
		src_files = os.listdir(dirFrom)
		for file_name in src_files:
			full_file_name = os.path.join(dirFrom, file_name)
			if (os.path.isfile(full_file_name)):
				shutil.copy(full_file_name, dirTo)
	except Exception as e:
		QIlogger.error("Impossible to copy files to new models " + str(e))


def replaceDataInCheckpoint(checkpointPath, userName):
	try:
		with open(checkpointPath, 'r') as myfile:
			data = myfile.read()

		output = re.sub("base_model",  userName, data)

		with open(checkpointPath, 'w', encoding="UTF-8") as f:
			f.write("%s\n" % output)

	except Exception as e:
		QIlogger.error("Impossible to replace Data in checkpoint " + str(e))

"""
####### NEW CUSTOMER SCRIPT #######
"""

def newCustomer(userName, userEmail, password, customerEmail, modelsDir) :
	try:
		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("1 - New Customer Script")

		#
		# Connector API
		configConnector = con.ConfigConnection()
		connector = con.Connector(configConnector)
		body_params = {
			"email": userEmail,
			"cellphone": "0000",
			"password": password,
			"emailsupport" : customerEmail
		}

		QIlogger.info("------------------------------------------------------")
		QIlogger.info("2 - SignIN new User")
		#
		#connector.signin(body_params)
		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("3 - Add User to script/config")
		#

		with open("config/config.json", 'r') as json_file:
			data = json.load(json_file)
			data["users"].append({
				'name' : userName,
				'config_dir' : ''
			})

		with open("config/config.json", 'w') as json_file:
			json.dump(data, json_file)
			json_file.close()



		QIlogger.info("------------------------------------------------------")
		QIlogger.info("4 - Create DIR main Config in ticket_classification")
		#
		userConfigDir = dir_path +'/../ticket_classification/config/' + userName
		createNewDir(userConfigDir)


		QIlogger.info("------------------------------------------------------")
		QIlogger.info("5 - Copy base models config and connector to new user DIR config")
		#
		copyDatasFromToDirs(dir_path + '/../base_configs', userConfigDir)

		QIlogger.info("------------------------------------------------------")
		QIlogger.info("6 - Read models config and change paths")
		#
		with open(userConfigDir + "/category_config.json", 'r') as json_file:
			dataC = json.load(json_file)
			dataC["paths"]["main_path"] = dataC["paths"]["main_path"].replace("base_config", userName)
			#json.dump(data, json_file)

		with open(userConfigDir + "/category_config.json", 'w') as json_file:
			json.dump(dataC, json_file)
			json_file.close()
		#
		with open(userConfigDir + "/priority_config.json", 'r') as json_file:
			dataP= json.load(json_file)
			dataP["paths"]["main_path"] = dataP["paths"]["main_path"].replace("base_config", userName)
			#json.dump(data, json_file)

		with open(userConfigDir + "/priority_config.json", 'w') as json_file:
			json.dump(dataP, json_file)
			json_file.close()

		#
		with open(userConfigDir + "/connector_config.json", 'r') as json_file:
			dataCC = json.load(json_file)
			dataCC["web_service"]["user"] = userEmail
			dataCC["web_service"]["password"] = password
			#json.dump(data, json_file)

		with open(userConfigDir + "/connector_config.json", 'w') as json_file:
			json.dump(dataCC, json_file)
			json_file.close()


		#
		QIlogger.info("------------------------------------------------------")
		QIlogger.info("7 - Create user main DIR models")
		#
		userModelDir = modelsDir + '/' + userName
		createNewDir(userModelDir)

		QIlogger.info("------------------------------------------------------")
		QIlogger.info("8 - Copy base models checkpoints and datas to new user DIR")
		#
		#copyDatasFromToDirs(modelsDir + '/base_model', userModelDir)
		copy_tree(modelsDir + '/base_model', userModelDir)

		QIlogger.info("------------------------------------------------------")
		QIlogger.info("9 - Change name in the checkpoint files")
		#
		replaceDataInCheckpoint(modelsDir + '/' + userName + '/models/category/model/best_models/checkpoint', userName)
		replaceDataInCheckpoint(modelsDir + '/' + userName + '/models/priority/model/best_models/checkpoint', userName)

		QIlogger.info("------------------------------------------------------")
		QIlogger.info("10 - Add a crontab")
		#
		my_cron = CronTab(user='questit')
		cronJob_command = '/user/bin/python3 /home/questit/assist_classifier/scripts/classification_script.py ' + userName + ' > /home/questit/tmp/classification_log_' + userName+ '.txt'
		job = my_cron.new(command=cronJob_command)
		job.minute.every(3)
		my_cron.write()

	except Exception as e:
		QIlogger.error("Error in New Customer Script method " + str(e))


"""
''' MAIN '''
"""
def main_createUser():
	if len(sys.argv) < 4:
		print("add userName, userEmail, password, customerEmail, modelsDir to run the script.")
		exit(1)
	#
	userName = sys.argv[1]
	userEmail = sys.argv[2]
	password = sys.argv[3]
	customerEmail = sys.argv[4]
	modelsDir = '/home/questit/data/models_and_data'
	#modelsDir = '/home/antonio/Scrivania/Lavoro/AssistProject/assist/data/models_and_data'
	lg.configureLogger(QIlogger, '', "create_User")
	newCustomer(userName, userEmail, password, customerEmail, modelsDir)



if __name__ == '__main__':
	#main()
	main_createUser()
