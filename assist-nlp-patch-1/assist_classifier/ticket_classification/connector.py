import requests
import json
import sys
import logger as lg
from requests.auth import HTTPDigestAuth
import time



QIlogger = lg.getLogger("__main__")

class ConfigConnection():
	def __init__(self):
		# Web Service
		self.web_service = {"url": "https://assist-prj.org/assist/api/v1", "user": "test@quest-it.com",
							"password": "prova", "token_id": "assist-sid", "numb_next_tickets" : 1000}
		# self.web_service = {"url": "http://httpbin.org/post", "user": "", "password": ""}
		self.web_service_params = {"": ""}
		self.max_tickets_to_get = 10000

	"""##########################################################################"""

	def loadConfigFile(self,filePath):
		with open(filePath) as f:
			data = json.load(f)
		return data

	"""##########################################################################"""

	def configFromFile(self, dataPath):
		data = self.loadConfigFile(dataPath)
		self.web_service = data['web_service']

	"""##########################################################################"""

	def getWebService(self):
		return self.web_service

class Connector(object):
	def __init__(self,config):
		self.web_service = config.web_service
		self.token = None
	"""#####################################################"""
	def getToken(self):
		return self.token

	"""#####################################################"""
	def setToken(self, response):
		cookies = response.cookies
		self.token = cookies[self.web_service["token_id"]]
		QIlogger.info("	-- Token Setted")

	"""#####################################################"""
	def checkStatus(self, response):
		m = ""
		status = response.status_code
		"""Check the returned status from the request"""
		if status == 200:
			QIlogger.info("	Status Code: 200 - OK")
		else :
			if status == 301:
				m = "Redirection to a different endpoint"
			if status == 401:
				m = "Not Authenticated"
			if status == 400:
				m = "Bad Request"
			if status == 403:
				m = "Forbidden"
			if status == 404:
				m = "Not Found"
				QIlogger.info("	Status Code: " + str(status) + " - Error! - " + str(m))
			raise QIlogger.error("	Status Code: " +str(status) + " - Error! - " + str(m))

	"""#####################################################"""
	def checkConnection(self):
		#response = requests.post(self.web_service["url"], auth=(self.web_service["user"], self.web_service["password"]))
		#print(response)
		response = requests.post(self.web_service["url"])
		status = response.status_code
		self.checkStatus(status)

	"""#####################################################"""
	def checkCookies(self, sess):
		cookies = sess.cookies
		if not cookies:
			raise Exception("Cookie empty!")
		else :
			return True

	"""#####################################################"""
	def checkToken(self, sess):
		cookies = sess.cookies
		if not cookies:
			raise Exception("Cookie empty!")
		else:
			if self.web_service["token_id"] in cookies.keys():
				if cookies[self.web_service["token_id"]] == "" or cookies[self.web_service["token_id"]] == None :
					raise QIlogger.error("Empty Token in Cookie")
				else:
					return True
			else:
				raise QIlogger.error("Token key not in Cookie Dict")

	"""#####################################################"""
	def call(self, call, params, auth, type):
		if type == "get":
			response = requests.get(self.web_service["url"] + "/" + call, params=params, auth=auth)
		else:
			response = requests.post(self.web_service["url"] + "/" + call, params=params, auth=auth)
		return response

	"""#####################################################"""
	def signin(self, body_params):
		"""Create a new user, body_params deve contenere i campi : email,cellphone,password,emailsupport (assist_support@assist_classifier-prj.org)"""
		try:
			response = requests.post(self.web_service["url"] + "/signin", json=body_params)
		except requests.exceptions.RequestException as e:
			print(e)
			sys.exit(1)
		# check status
		self.checkStatus(response)

	"""#####################################################"""
	def login(self, sess):
		QIlogger.info("--------------------------------")
		QIlogger.info("========= LOGIN ================")
		data = {
			'username' : self.web_service["user"],
			'password' : self.web_service["password"]
		}
		response = None
		#response
		try:
			QIlogger.info("	-- Login to API URL : " + str(self.web_service["url"] + "/login"))
			response = sess.post(self.web_service["url"] + "/login", json=data)
		except requests.exceptions.Timeout:
			QIlogger.info("	-- Error on Request : TimeOut")
			for i in range(0,5):
				QIlogger.info("	-- New Try : " + str(i))
				response = sess.post(self.web_service["url"] + "/login", json=data)
				if response.status_code == 200:
					break
		except requests.exceptions.TooManyRedirects:
			raise QIlogger.error("	-- Error on Request: Too many Redirects")
		except requests.exceptions.RequestException as e:
			print(e)
			sys.exit(1)
		#check status
		QIlogger.info("	-- Checking Status")
		self.checkStatus(response)
		#check cookie
		QIlogger.info("	-- Checking Cookies")
		self.checkCookies(sess)
		#check token
		QIlogger.info("	-- Checking Token")
		self.checkToken(sess)
		#set token
		QIlogger.info("	-- Set Token")
		self.setToken(response)
		QIlogger.info("========= LOGGED IN ================")

	"""#####################################################"""
	def logout(self, sess):
		QIlogger.info("--------------------------------")
		QIlogger.info("========= LOGOUT ================")
		response = None
		try:
			response = sess.post(self.web_service["url"] + "/logout")
		except requests.exceptions.Timeout:
			QIlogger.info("	-- Error on Request : TimeOut")
			for i in range(0,5):
				QIlogger.info("	-- New Try : " + str(i))
				response = sess.post(self.web_service["url"] + "/logout")
				if response.status_code == 200:
					break
		except requests.exceptions.TooManyRedirects:
			raise QIlogger.error("	-- Error on Request: Too many Redirects")
		except requests.exceptions.RequestException as e:
			print(e)
			sys.exit(1)
		# check status
		QIlogger.info("	-- Checking Status")
		self.checkStatus(response)
		QIlogger.info("	-- Reset Token")
		self.token = None
		QIlogger.info("========= LOGGED OUT ================")

	"""#####################################################"""
	def getNextTicket(self,sess):
		QIlogger.info("------------------------------------------")
		QIlogger.info("========= GET NEXT TICKET ================")
		response = None
		try:
			response = sess.get(self.web_service["url"] + "/tickets/next")
		except requests.exceptions.Timeout:
			QIlogger.info("	-- Error on Request : TimeOut")
			for i in range(0, 5):
				QIlogger.info("	-- New Try : " + str(i))
				response = sess.get(self.web_service["url"] + "/tickets/next")
				if response.status_code == 200:
					break
		except requests.exceptions.TooManyRedirects:
			raise QIlogger.error("	-- Error on Request: Too many Redirects")
		except requests.exceptions.RequestException as e:
			print(e)
			sys.exit(1)
		# check status
		QIlogger.info("	-- Checking Status\n")
		self.checkStatus(response)
		QIlogger.info("-Response " + str(response.json()))
		QIlogger.info("------------------------------------------")
		return response.json()

	"""#####################################################"""
	def sendClassification(self, sess, ticket_id, params):
		"""Send to the API the classification of the ticket (category and priority). Struttura di params={categoryId:number,priorityId:number}"""
		QIlogger.info("------------------------------------------")
		QIlogger.info("========= SEND CLASSIFICATION START ============")
		QIlogger.info("----- Ticket id : " + ticket_id + " - Params : " + str(params) + " -----")
		response = None
		try:
			response = sess.put(self.web_service["url"] + "/tickets/" + ticket_id + "/category", json=params)
		except requests.exceptions.Timeout:
			QIlogger.info("	-- Error on Request : TimeOut\n")
			for i in range(0, 5):
				QIlogger.info("	-- New Try : " + str(i))
				response = sess.put(self.web_service["url"] + "/tickets/" + ticket_id + "/category", json=params)
				if response.status_code == 200:
					break
		except requests.exceptions.TooManyRedirects:
			raise QIlogger.error("	-- Error on Request: Too many Redirects")
		except requests.exceptions.RequestException as e:
			QIlogger.info(e)
			sys.exit(1)

		# check status
		QIlogger.info("	-- Checking Status")
		self.checkStatus(response)
		QIlogger.info("========= SEND CLASSIFICATION END ============")
		QIlogger.info("------------------------------------------")

	"""#####################################################"""
	def getTickets(self,sess,params):
		"""Get all closed Tickets between two dates, params = closedfrom:date&closedto:date&maxnum:integer as a dict()"""
		QIlogger.info("------------------------------------------")
		QIlogger.info("========= GET TICKETS START ============")
		response = None
		try:
			response = sess.get(self.web_service["url"] + "/tickets/tickets?", params=params)
		except requests.exceptions.Timeout:
			QIlogger.info("	-- Error on Request : TimeOut")
			for i in range(0, 5):
				QIlogger.info("	-- New Try : " + str(i))
				response = sess.get(self.web_service["url"] + "/tickets/tickets?", params=params)
				if response.status_code == 200:
					break
		except requests.exceptions.TooManyRedirects:
			raise QIlogger.error("	-- Error on Request: Too many Redirects")
		except requests.exceptions.RequestException as e:
			QIlogger.info(e)
			sys.exit(1)

		# check status
		QIlogger.info("	-- Checking Status")
		self.checkStatus(response)
		descJson = json.loads(response.text)
		QIlogger.info("========= GET TICKETS END ============")
		QIlogger.info("------------------------------------------")
		return descJson

def connectToService():
	config = ConfigConnection()
	connector = Connector(config)
	signin_params = {"email": "", "cellphone" : " ", "password" : "", "emailsupport" : "assist_support@assist_classifier-prj.org"}
	#connector.signin(signin_params)
	#Create Persisten Session
	sess = requests.session()
	#LogIN
	connector.login(sess)
	#response = connector.getNextTicket(sess)
	#LogOut
	connector.logout(sess)


#connectToService()
