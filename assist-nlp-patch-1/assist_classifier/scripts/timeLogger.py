import os
import sys
import json
import datetime

class TimeLogger(object):
	def __init__(self, timeLoggerFile):
		self.file = timeLoggerFile
	#
	def createTimeFile(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/timeLogger/" + self.file

		if os.path.isfile(pidfile):
			print("%s already exists, exiting " % pidfile)
			sys.exit()

		data = {}
		data['last_date'] = []
		data['last_date'].append({'date' : datetime.datetime.today().strftime('%Y-%m-%d')})

		with open(pidfile, 'w') as outfile:
			json.dump(data, outfile)
	#
	def getLastDate(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/timeLogger/" + self.file

		if os.path.isfile(pidfile):
			with open(pidfile) as json_file:
				data = json.load(json_file)
				for p in data['last_date']:
					return p['date']
		else:
			raise Exception("last date not found! ")
	#
	def deleteFile(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/timeLogger/" + self.file

		if os.path.isfile(pidfile):
			os.unlink(pidfile)

	def checkFileExistence(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/timeLogger/" + self.file

		if os.path.isfile(pidfile):
			return True
		else:
			return False