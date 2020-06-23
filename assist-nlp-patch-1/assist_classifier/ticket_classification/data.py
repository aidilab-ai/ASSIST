import csv
from collections import defaultdict
import numpy as np
import re
from utility import deprecated
import json
import codecs

"""##################################################################"""
"""DataLoader"""
"""##################################################################"""
class Data(object):
	def __init__(self, config):
		self.config = config

	"""##################################################################"""

	def load_data(self):
		"""Load data (only Aperture with Chiusure Label)"""
		first_level_targets = []
		first_level_labels = []
		tickets = []
		complete_targets = []
		complete_labels = []

		columns = defaultdict(list)
		#
		with codecs.open(self.config.csv_path, 'r', encoding = self.config.op_encoding, errors ='replace') as csvfile:
			csv_parsed = []
			reader = csvfile.read().split("\n")
			count = 0
			for line in reader:
				line = line.encode('utf-8').decode('utf-8')
				if line == reader[0]:
					head = line
					continue
				if re.match("\d+\|\d+\|\|(.+)", line.strip()):
					#print(str(line))
					m =  line
					csv_parsed.append(m)
					count = count +1
				else:
					previous_line = csv_parsed[count-1]
					new_line = previous_line + " " + line
					csv_parsed.pop(count-1)
					csv_parsed.append(new_line)
					count = len(csv_parsed)

			new_csv_parsed = []
			for line in csv_parsed:
				nl=line.replace("\n", " ")
				new_csv_parsed.append(nl)

			previous_ticket = 0
			flag_apertura = False
			for line in new_csv_parsed:
				#get id_ticket
				cc = re.split("\|", line)
				current_ticket = cc[0]
				if current_ticket == previous_ticket:
					search = "Chiusura"
				else :
					flag_apertura = False
					previous_ticket = current_ticket
					search = "Apertura"

				#get complete target
				strim_begin = re.split("\d+\|\d+\|(.+)\|", line)
				target = re.split("\|", strim_begin[1])
				#complete_targets.append(target[1])
				if target[1] not in complete_labels:
					complete_labels.append(target[1])
				#get 1° level of target
				target_1_level = target[1].split(" ",1)
				first_level_targets.append(target_1_level[0])
				if target_1_level[0] not in first_level_labels:
					first_level_labels.append(target_1_level[0])

				#cut_date_split = strim_begin[1].rsplit("|",1)
				#cut_date = cut_date_split[0]
				cut_date = strim_begin[1]
				#get ticket
				ticket_split = re.split("\|",cut_date)
				stato_ticket = ticket_split[2]
				if stato_ticket == "Apertura" and search =="Apertura"  or stato_ticket =="Apertura ticket fornitore" and search =="Apertura" or stato_ticket =="Apertura Attività" and search =="Apertura":
					ticket = ticket_split[3].rstrip('\r\n').replace('\n', ' ').replace('\r', ' ')
					if ticket != " " and ticket != "" and ticket != None:
						tickets.append(ticket)
						flag_apertura = True
						complete_targets.append(target[1])
				if stato_ticket == "Chiusura" and search == "Chiusura" and flag_apertura == True or stato_ticket =="Chiusura Attività" and search =="Chiusura":
					complete_targets = complete_targets[:-1]
					complete_targets.append(target[1])
					previous_ticket = 0
					flag_apertura = False
				#else:
					#complete_targets = complete_targets[:-1]

			if len(complete_targets) != len(tickets) :
				raise Exception('Number of targets and number of inputs are different')

		return complete_targets, tickets, complete_labels, first_level_targets, first_level_labels

	"""##################################################################"""
	def removeRiassociazione(self, datapath):
		#
		with codecs.open(self.config.csv_path, 'r', encoding=self.config.op_encoding, errors='replace') as csvfile:
			csv_parsed = []
			reader = csvfile.read().split("\n")
			count = 0
			for line in reader:
				line = line.encode('utf-8').decode('utf-8')
				if line == reader[0]:
					head = line
					continue
				if re.match("\d+\|\d+\|\|(.+)", line.strip()):
					#print(str(line))
					m =  line
					csv_parsed.append(m)
					count = count +1
				else:
					previous_line = csv_parsed[count-1]
					new_line = previous_line + " " + line
					csv_parsed.pop(count-1)
					csv_parsed.append(new_line)
					count = len(csv_parsed)

			new_csv_parsed = []
			for line in csv_parsed:
				nl = line.replace("\n", " ")
				new_csv_parsed.append(nl)

		return None
	"""##################################################################"""
	@deprecated("Not in use")
	def createHierarchicalLabel(self, first_level_labels, complete_labels):
		hierarchy = {}
		for id, label in enumerate(first_level_labels) :
			hierarchy[label] = {"id" : id, "level" : 1, "parent" : None, "children" : []}

		for n in complete_labels :
			label_split = n.split(" ", 1)
			for i in hierarchy :
				if label_split[0] == i and len(label_split) > 1 :
					if label_split[1] not in hierarchy[i]["children"] :
						hierarchy[i]["children"].append(label_split[1])
		return hierarchy

	"""##################################################################"""

	def writeArrayInFile(self, data, filename, encoding):
		with open(self.config.main_path + '/' + filename, 'w', encoding=encoding) as f:
			for item in data:
				f.write(item + "\n")
				#f.write(item)

	def writeArrayInFileCompleteDataPath(self, data, datapath, encoding):
		with open(datapath, 'w', encoding=encoding) as f:
			for item in data:
				f.write("%s\n" % item)
				#f.write(item + "\n")
				#f.write(item)

	def writeArrayStringInFile(self, data, filename, encoding):
		with open(self.config.main_path + '/' + filename, 'w', encoding=encoding) as f:
			for item in data:
				f.write("%s\n" % item)

	"""##################################################################"""

	def loadDataInArray(self,datapath, encoding="utf-8"):
		lines = []
		with open(datapath, encoding=encoding) as f:
			for line in f :
				line = line.encode('utf-8').decode('utf-8')
				line = line.strip()
				line = line.rstrip()
				line = line.rstrip('\n')
				lines.append(line)
			#lines = f.read().splitlines()
		return lines

	"""##################################################################"""

	def loadMapFromJson(self,jsonpath):
		with open(jsonpath, encoding='utf-8') as data_file:
			data = json.loads(data_file.read())
		return data


	"""##################################################################"""

	def mapTargets(self,map,targets):
		def get_parent(map,target, level_zero_name=''):
			for element in map:
				if element['level'] == 0:
					level_zero_name = element['name']

				if element['name'] == target:
					if element['level'] == 0:
						return target
					else:
						return level_zero_name
				else:
					if element['children'] and len(element['children']) > 0:
						children_len = len(element['children'])
						node_name = get_parent(element['children'], target, level_zero_name)
						if node_name != None:
							return node_name
					else :
						continue
		t_zeros = []
		for target in targets:
			name = get_parent(map, target)
			if name != None:
				t_zeros.append(name)
			else :
				t_zeros.append("__" + target + "__")

		return t_zeros

	"""##################################################################"""

	def getfirstLevelTargets(self,map):
		mainLabels=[]
		for element in map:
			if element['level'] == 0:
				level_zero_name = element['name']
				mainLabels.append(level_zero_name)
		return mainLabels

	"""##################################################################"""

	def createDataSequence(self,tickets,dictionary):
		padded_datasets_ids = []
		for ticket in tickets:
			ticket_ids = []
			#ticket_ids = [dictionary[w] if w in dictionary else dictionary["UNK"] for w in ticket]
			for w in ticket:
				if re.match('euro',w) or re.match('€',w):
					ticket_ids.append(dictionary[self.config.currency_token])
				elif w.isdigit() or re.match('(\d+.\d+ | \d+ | \d+.\d+,\d+ | \d+,\d+)', w):
					ticket_ids.append(dictionary[self.config.numeric_token])
				elif re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3}|[01]\d)',w):
					ticket_ids.append(dictionary[self.config.date_token])
				elif w in dictionary:
					ticket_ids.append(dictionary[w])
				else:
					ticket_ids.append(dictionary[self.config.unkown_token])


			#
			def pad_list(l, pad_token, max_l_size):
				padded_l = []

				max_l = min(max_l_size, len(l))
				for i in range(max_l):
					padded_l.append(l[i])
				#
				for j in range(len(l), max_l_size):
					padded_l.append(pad_token)
				#
				return padded_l

			#
			pad_id = dictionary[self.config.pad_token]
			padded_data = pad_list(ticket_ids, pad_id, self.config.max_length_sequence)
			padded_datasets_ids.append(padded_data)

		return padded_datasets_ids

	"""##################################################################"""

	def createDataSequenceTicketsString(self,tickets,dictionary):
		padded_datasets_ids = []
		for ticket in tickets:
			ticket_ids = []
			#ticket_ids = [dictionary[w] if w in dictionary else dictionary["UNK"] for w in ticket]
			ticket_splitted = ticket.split(" ")
			for w in ticket_splitted:
				if re.match('euro',w) or re.match('€',w):
					ticket_ids.append(dictionary[self.config.currency_token])
				elif w.isdigit() or re.match('(\d+.\d+ | \d+ | \d+.\d+,\d+ | \d+,\d+)', w):
					ticket_ids.append(dictionary[self.config.numeric_token])
				elif re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3}|[01]\d)',w):
					ticket_ids.append(dictionary[self.config.date_token])
				elif w in dictionary:
					ticket_ids.append(dictionary[w])
				else:
					ticket_ids.append(dictionary[self.config.unkown_token])


			#
			def pad_list(l, pad_token, max_l_size):
				padded_l = []

				max_l = min(max_l_size, len(l))
				for i in range(max_l):
					padded_l.append(l[i])
				#
				for j in range(len(l), max_l_size):
					padded_l.append(pad_token)
				#
				return padded_l

			#
			pad_id = dictionary[self.config.pad_token]
			padded_data = pad_list(ticket_ids, pad_id, self.config.max_length_sequence)
			padded_datasets_ids.append(padded_data)

		return padded_datasets_ids

	"""##################################################################"""

	def fromSequenceStringToSequenceArray(self, datas):
		tickets_parsed = []
		for t in datas:
			tickets_work = []
			tt = re.split("\[", t)
			tt = re.split("\]", tt[1])
			tt = re.split(",", tt[0])
			for inner_t in tt:
				a = int(inner_t)
				tickets_work.append(a)
			tickets_parsed.append(tickets_work)
		return tickets_parsed

	def fromSequenceTokensToSequenceTokenArray(self, datas):
		tickets_parsed = []
		for t in datas:
			tickets_work = []
			tt = re.split("\[", t)
			tt = re.split("\]", tt[1])
			tt = re.split(",", tt[0])
			for inner_t in tt:
				ff = re.split("'", inner_t)
				tickets_work.append(ff[1])
			tickets_parsed.append(tickets_work)
		return tickets_parsed

	"""##################################################################"""

	def fromSequenceToData(self,ticket,dictionary):
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		sequenceString = ""
		for t in ticket:
			st = reverse_dictionary[t]
			sequenceString = sequenceString + " " + st
		return sequenceString

	"""##################################################################"""

	def fromDataToBagOfWords(self,data,dictionary):
		len_dict = len(dictionary)
		tickets_bag = []

		for ticket in data:
			ticket_bag = np.zeros((len_dict,), dtype=int)
			ticket_tokens = re.split(" ", ticket)
			for w in ticket_tokens:
				ticket_ids = 0
				if re.match('euro',w) or re.match('€',w):
					ticket_ids = dictionary[self.config.currency_token]
				elif w.isdigit() or re.match('(\d+.\d+ | \d+ | \d+.\d+,\d+ | \d+,\d+)', w):
					ticket_ids = dictionary[self.config.numeric_token]
				elif re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3}|[01]\d)',w):
					ticket_ids = dictionary[self.config.date_token]
				elif w in dictionary:
					ticket_ids = dictionary[w]
				else:
					ticket_ids = dictionary[self.config.unkown_token]
				ticket_bag[ticket_ids] = 1
			tickets_bag.append(ticket_bag)
		return tickets_bag

	"""##################################################################"""

	def __transformInOneHotVector(self,mapping,data):
		dict_map = dict()
		dict_map.update([(m,i) for i,m in enumerate(mapping)])
		vectors = []
		i = 0
		for d in data:
			oneHot = np.zeros((len(mapping),), dtype=int)
			indexM = dict_map[d]
			if not indexM >= 0 and not indexM <= (len(mapping) - 1):
				print("d : " + str(d) + " | posizione : " +str(i))
			oneHot[indexM] = 1
			vectors.append(oneHot)
			i = i + 1
		vectors = np.asarray(vectors)
		return vectors

	"""##################################################################"""

	def transformInOneHotVector(self,mapping,data):
		dict_map = dict()
		dict_map.update([(m,i) for i,m in enumerate(mapping)])
		vectors = []
		i = 0
		for d in data:
			try:
				oneHot = np.zeros((len(mapping),), dtype=int)
				indexM = dict_map[d]
				if not indexM >= 0 and not indexM <= (len(mapping) - 1):
					print("d : " + str(d) + " | posizione : " +str(i))
				oneHot[indexM] = 1
				vectors.append(oneHot)
			except Exception as e:
				oneHot = np.zeros((len(mapping),), dtype=int)
				oneHot[0] = 1
				vectors.append(oneHot)
				print("exception at target line " + str(i) + " Data  " + str(d) + " . Exception -- " + str(e))
			i = i + 1
		vectors = np.asarray(vectors)
		return vectors

	"""##################################################################"""

	def transformSingleTargetInOneHotVector(self,mapping,data):
		dict_map = dict()
		dict_map.update([(m,i) for i,m in enumerate(mapping)])
		vectors = []

		oneHot = np.zeros((5,), dtype=int)
		indexM = dict_map[data]
		if indexM >= 0 and indexM <= 4:
			oneHot[indexM] = 1

		vector = np.asarray(oneHot)
		return vector

	"""##################################################################"""

	def transformListStringInOneHot(self,data):
		vectors = []
		for d in data:
			arr_d = re.split("\[", d)
			arr_d = re.split("\]", arr_d[1])
			arr_d = re.split(" ", arr_d[0])
			count = 0
			for ad, in arr_d :
				if ad == "1" :
					break
				else :
					count = count + 1
			oneHot = np.zeros((5,), dtype=int)
			oneHot[count] = 1
			vectors.append(oneHot)

		vectors = np.asarray(vectors)
		return vectors

	"""##################################################################"""

	def fromOneHotToTerm(self,mapping,vector):
		dict_map = dict()
		dict_map.update([(i,m) for i,m in enumerate(mapping)])
		el = np.where(vector == 1)
		mapped = dict_map[el[0][0]]
		return mapped

	"""##################################################################"""

	def overSampleData(self, tickets, targets, labels, classToEnance, mul):
		tickets_to_append = []
		targets_to_append = []
		classesToEnance = []

		class_converter = self.transformSingleTargetInOneHotVector(labels, classToEnance)
		classesToEnance.append(class_converter)

		for j in range(mul):
			for i in range(len(targets)):
				if targets[i] in classToEnance:
					tickets_to_append.append(tickets[i])
					targets_to_append.append(targets[i])

		tickets.extend(tickets_to_append)
		targets.extend(targets_to_append)

		return tickets, targets

	"""##################################################################"""

	def countClassOccurrences(self, targets, labels, type):
		targetsRemapped = []
		#for i in range(len(targets)):
			#cl = ut.fromOneHotToTerm(labels, targets[i])
			#targetsRemapped.append(cl)

		for label in labels:
			counter = targets.count(label)
			print( "	" + str(type) + " | Number of data in class " + str(label) + " : " + str(counter))
		print("\n")

	"""##################################################################"""

	def extractFeatures(self, tickets, dictionary):
		features = self.config.features_to_extract
		sequences_features = []
		for ticket in tickets:
			ticket_features = []
			for w in ticket:
				word_feature = np.zeros((len(features),), dtype=int)
				if "check_isINVoc" in features:
					if w in dictionary:
						word_feature[features.index("check_isINVoc")] = 1
				if "check_isNumeric" in features:
					if w.isdigit() or re.match('(\d+.\d+ | \d+ | \d+.\d+,\d+ | \d+,\d+)', w):
						word_feature[features.index("check_isNumeric")] = 1
				if "check_isCurrency" in features:
					if re.match('euro', w) or re.match('€', w):
						word_feature[features.index("check_isCurrency")] = 1
				if "check_isDate" in features:
					if re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3}|[01]\d)',w):
						word_feature[features.index("check_isDate")] = 1

				ticket_features.append(word_feature.tolist())
			#add pad or remove features if sequence is too long
			if len(ticket_features) < self.config.max_length_sequence:
				num_pad_sequence = self.config.max_length_sequence - len(ticket_features)
				for i in range(num_pad_sequence):
					empty_feature = np.zeros((len(features),), dtype=int)
					ticket_features.append(empty_feature)
			else :
				ticket_features = ticket_features[0:self.config.max_length_sequence]

			sequences_features.append(ticket_features)
		return sequences_features

	"""##################################################################"""

	def generatePriorityTarget(self,tickets):
		targets = ["1"] * len(tickets)
		for i in range(len(tickets)):
			if "chiede" in tickets[i]:
				targets[i] = "2"
			if "fattura" in tickets[i] or "segnala" in tickets[i]:
				targets[i] = "3"
			if "sollecita" in tickets[i] or "guasto" in tickets[i]:
				targets[i] = "4"
			if "urgente" in tickets[i]:
				targets[i] = "5"
		return targets


	def __generatePriorityTarget(self, tickets):
		targets = ["1"] * len(tickets)
		for i in range(len(tickets)):
			if "chiede" in tickets[i] or "contattare" in tickets[i] or "ricontattare" in tickets[i] or "ricontattato" in tickets[i] or "ricontattata" in tickets[i] or "appuntamento" in tickets[i]:
				targets[i] = "2"
			if "fattura" in tickets[i] or "segnala" in tickets[i] or "disdire" in tickets[i] or "disdice" in tickets[i] or "disdetto" in tickets[i]:
				targets[i] = "3"
			if "sollecita" in tickets[i] or "guasto" in tickets[i] or "errore" in tickets[i]:
				targets[i] = "4"
			if "urgente" in tickets[i] or "urgenza" in tickets[i]:
				targets[i] = "5"

		return targets




def main():
	config = ""
	data = Data(config)
	tickets = data.loadDataInArray("C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/priority/model/data/tickets.txt", "UTF-8")
	targets = data.generatePriorityTarget(tickets)
	data.writeArrayInFileCompleteDataPath(targets,'C:/Users/Antonio/Desktop/Lavoro/assist_ticket_data/antonio/models/priority/model/data/targets.txt', "utf-8")

#main()
