import collections
import json

class Vocabulary(object):
	def __init__(self, config):
		self.config = config


	"""#############################################################################################################"""
	def build_dictionary(self,words,labels):
		count = [[self.config.unkown_token,-1],[self.config.numeric_token,-1],[self.config.date_token, -1],[self.config.currency_token, -1],[self.config.pad_token, -1]]
		count.extend(collections.Counter(words).most_common(self.config.vocab_size-6))

		dictionary = dict()
		for word,_ in count:
			dictionary[word] = len(dictionary)
		data = list()
		unk_count = 0
		for word in words:
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0
				unk_count+=1
			data.append(index)
		#
		count[0][1] = unk_count
		#
		for label in labels:
			dictionary[label] = len(dictionary)
		#
		reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
		#
		return dictionary,reverse_dictionary


	def saveDictionary(self, data, dataName):
		with open(self.config.data_path+'/'+dataName, 'w',  encoding=self.config.csv_encoding) as file:
			file.write(str(data))

	def loadDictionary(self,dataName):
		data = eval(open(self.config.data_path + '/' + dataName, encoding=self.config.csv_encoding).read())
		return data

	def loadDictionaryFromPath(self,dataPath):
		data = eval(open(dataPath, encoding=self.config.csv_encoding).read())
		return data

	def getReverseDictionary(self,dictionary):
		#
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return reverse_dictionary