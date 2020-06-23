import logging
import logging.handlers as handlers
import os

##########################################################
def getLogger(name):
	return logging.getLogger(name)


##########################################################
def configureLogger(logger, customer, mode):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	logCustomer = dir_path + '/log/log_' + mode + '_' + customer + '.log'
	logger.setLevel(logging.INFO)
	#FileHandler
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logHandler = handlers.RotatingFileHandler(logCustomer, maxBytes=100000, backupCount=5)
	logHandler.setLevel(logging.INFO)
	logHandler.setFormatter(formatter)
	logger.addHandler(logHandler)
	# ConsoleHandler
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)


##########################################################
def removeHandlers(logger):
	for handler in logger.handlers[:] :
		logger.removeHandler(handler)

