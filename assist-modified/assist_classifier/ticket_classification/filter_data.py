import utility as ut
import tokenizer as tk

class FilterData(object):
	def __init__(self, config, labels):
		self.config = config
		self.labels = labels

	def balanceData(self, tickets, targets, dimension):
		tickets, targets = ut.balanceSubSamplesReturnArray(tickets, targets, self.labels, dimension)
		return tickets, targets

	def removeIdenticalData(self, tickets, targets):
		tickets, targets = ut.removeIdenticalTickets(tickets, targets)
		return tickets, targets

	def tokenizeData(self, tickets):
		# Tokenize data
		tok = tk.Tokenizer(tickets)
		tok.tokenizeTickets()
		tickets_to_lower = tok.toLower()
		return tickets_to_lower

	def removeStopWordsFromData(self, tickets, targets):
		# Tokenize data
		tok = tk.Tokenizer(tickets)
		tickets_no_sw, targets_no_sw = tok.removeStopWords(tickets, targets)
		# create the array of words from all the tickets
		words = tok.extractWords()

		return tickets_no_sw, targets_no_sw, words

	def removeTokenOOV(self,tickets_seq,targets_seq,vocabulary):
		trash = []
		len_sequence = self.config.max_length_sequence
		pad_id = vocabulary[self.config.pad_token]
		unk_id = vocabulary[self.config.unkown_token]
		new_tickets_seq = []
		new_targets_seq = []
		for i in range(len(tickets_seq)):
			count_pad = tickets_seq[i].count(pad_id)
			count_unk = tickets_seq[i].count(unk_id)
			voc_tokens = len_sequence - count_pad - count_unk
			if voc_tokens >= self.config.threshold_len:
				new_tickets_seq.append(tickets_seq[i])
				new_targets_seq.append(targets_seq[i])
			else:
				trash.append(tickets_seq[i])

		return new_tickets_seq, new_targets_seq, trash

	def trashingTicket(self, ticket_seq, vocabulary):
		len_sequence = self.config.max_length_sequence
		pad_id = vocabulary[self.config.pad_token]
		unk_id = vocabulary[self.config.unkown_token]
		count_pad = ticket_seq.count(pad_id)
		count_unk = ticket_seq.count(unk_id)
		voc_tokens = len_sequence - count_pad - count_unk
		if voc_tokens >= self.config.threshold_len:
			return False
		else:
			return True


	def removeTokenOOVwithSequenceFeatures(self,tickets_seq,targets_seq,tickets_feature,vocabulary):
		trash = []
		len_sequence = self.config.max_length_sequence
		pad_id = vocabulary[self.config.pad_token]
		unk_id = vocabulary[self.config.unkown_token]
		new_tickets_seq = []
		new_targets_seq = []
		new_tickets_feature_seq = []
		for i in range(len(tickets_seq)):
			count_pad = tickets_seq[i].count(pad_id)
			count_unk = tickets_seq[i].count(unk_id)
			voc_tokens = len_sequence - count_pad - count_unk
			if voc_tokens >= self.config.threshold_len:
				new_tickets_seq.append(tickets_seq[i])
				new_targets_seq.append(targets_seq[i])
				new_tickets_feature_seq.append(tickets_feature[i])
			else:
				trash.append(tickets_seq[i])

		return new_tickets_seq, new_targets_seq, new_tickets_feature_seq, trash

	def filterOutClassFromData(self, tickets, targets, classToFilterOut):
		new_tickets = []
		new_targets = []
		for i in range(len(targets)):
			if targets[i] not in classToFilterOut:
				new_targets.append(targets[i])
				new_tickets.append(tickets[i])
		return new_tickets, new_targets

	def filterOutTickets(self, tickets, targets):
		new_tickets=[]
		new_targets=[]
		for i in range(len(targets)):
			if targets[i] == 'CS' or targets[i] == 'CSA' or targets[i] == 'CSA 2° Livello' or targets[i] == 'IT' or targets[i] == 'Installazione' or targets[i] == 'PC/Stampante' or targets[i] == 'Reseller' or targets[i] == 'Marketing' or targets[i] == 'Accessi Archimede' or targets[i] == 'Inserimento Agenti' or targets[i] == 'Inserimento Eni/Fastweb' or targets[i] == 'Preventivi' or targets[i] == 'Attivazioni AV2' or targets[i] == 'Sovrascrittura':
				continue
			else:
				tic = tickets[i].find("Riassociato da")
				if tic != -1:
					ts = tickets[i].split("-",1)
					if len(ts) > 1:
						new_tickets.append(ts[1].rstrip())
					else :
						new_tickets.append("")
				else :
					new_tickets.append(tickets[i].rstrip())
				new_targets.append(targets[i])

		return new_tickets, new_targets

	def filterOutTicketsWithTargetsMap(self, map, targets, tickets):
		new_targets = []
		new_tickets = []

		def cycleOnMap(map, target, new_tickets, new_targets,m_map):
			for j in range(len(map)):

				if map[j]['name'] == target and map[j]['delete'] == True:
					new_tickets = new_tickets[:-1]
					new_targets = new_targets[:-1]
					return new_tickets, new_targets
				if map[j]['children'] and len(map[j]['children']) > 0:
					new_tickets, new_targets = cycleOnMap(map[j]['children'], target, new_tickets, new_targets, m_map)
				else:
					del m_map[0]
					new_tickets, new_targets = cycleOnMap(m_map, target, new_tickets, new_targets, m_map)
				return new_tickets, new_targets

		for i in range(len(targets)):
			new_targets.append(targets[i])
			new_tickets.append(tickets[i])
			new_tickets, new_targets = cycleOnMap(map, targets[i], new_tickets, new_targets, map)

		return new_tickets, new_targets
