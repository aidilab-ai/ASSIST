import json

"""##################################################################"""
"""Config"""
"""##################################################################"""
class Config():
	def __init__(self):
		#
		self.home_path = ""
		self.code_path = ""
		self.users = ""
		self.model_types = ""
		self.skipgram_model_path = ""

	"""##########################################################################"""

	def loadConfigFile(self, filePath):
		with open(filePath) as f:
			data = json.load(f)
		return data

	"""##########################################################################"""

	def mapMain2Config(self, dataJson):
		main = dataJson["main"]
		self.home_path = main["home_path"]
		self.code_path = main["code_path"]
		self.skipgram_model_path = main["skipgram_model_path"]

	"""##########################################################################"""

	def mapUsers2Config(self, dataJson):
		users = dataJson["users"]
		self.users = users

	"""##########################################################################"""

	def mapModelTypes(self, dataJson):
		modelType = dataJson["model_types"]
		self.model_types = modelType

	"""##########################################################################"""

	def configFromFile(self, dataPath):
		data = self.loadConfigFile(dataPath)
		self.mapMain2Config(data)
		self.mapUsers2Config(data)
		self.mapModelTypes(data)