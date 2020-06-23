import os
import sys
import platform
import time

class Demon(object):
	def __init__(self, pidFile):
		self.pidFile = pidFile
		self.pid = ""
	#
	def createPidDemon(self):
		pid = str(os.getpid())
		self.pid = pid
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/demon/" + self.pidFile

		if os.path.isfile(pidfile):
			print("%s already exists, exiting " % pidfile)
			sys.exit()
		open(pidfile, 'w+').write(pid)
	#
	def checkPidDemon(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/demon/" + self.pidFile

		if os.path.isfile(pidfile):
			return True
		else:
			return False
	#
	def deletePidDemon(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/demon/" + self.pidFile

		if os.path.isfile(pidfile):
			os.unlink(pidfile)

	#
	def getCreationTime(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		pidfile = dir_path + "/demon/" + self.pidFile
		timeStamp = ""
		if platform.system() == 'Windows':
			timeStamp = os.path.getctime(pidfile)
		else:

			stat = os.stat(pidfile)
			try:
				timeStamp = stat.st_birthtime
			except AttributeError:
				# We're probably on Linux. No easy way to get creation dates here,
				# so we'll settle for when its content was last modified.
				timeStamp = stat.st_mtime
		return int(timeStamp)


	def getCurrentTimeStamp(self):
		ts = int(time.time())
		return ts

	def findDaysBetweenTimeStamps(self, t1, t2):
		tfinal = t1 - t2
		tfinal = tfinal / 3600
		tfinal = tfinal / 24

		return tfinal



#main
def main():
	trainingDemon = Demon("training.pid")
	if trainingDemon.checkPidDemon() :
		ctime = trainingDemon.getCreationTime()
		today = trainingDemon.getCurrentTimeStamp()
		days = trainingDemon.findDaysBetweenTimeStamps(today,ctime)
		#
		#trainingDemon.createPidDemon()

#main()
