import nltk as nl
from nltk.corpus import stopwords
import re

"""##################################################################"""
"""Tokenizer"""
"""##################################################################"""
class Tokenizer():
	def __init__(self,data):
		self.data = data

	"""##################################################################"""
	def extractWords(self):
		words = []
		for line in self.data:
			print("line\n")
			print(line)
			#tokenized_verse = nl.word_tokenize(ll, language='italian')
			#tokenized_verse_l = re.split(" ", line)
			#tokenized_verse = [w.lower() for w in tokenized_verse if "\"" not in w]
			#remove puntaction
			#tt = []
			#for l in line :
				#tt.append(l.replace('[^\w\s]', ''))
			tokenized_verse = [w.lower() for w in line if "\"" not in w]
			tokenized_verse = [w for w in tokenized_verse if "," not in w]
			tokenized_verse = [w for w in tokenized_verse if ";" not in w]
			tokenized_verse = [w for w in tokenized_verse if ")" not in w]
			tokenized_verse = [w for w in tokenized_verse if "(" not in w]
			tokenized_verse = [w for w in tokenized_verse if "." not in w]
			tokenized_verse = [w for w in tokenized_verse if "\'" not in w]
			tokenized_verse = [w for w in tokenized_verse if ":" not in w]
			tokenized_verse = [w for w in tokenized_verse if "?" not in w]
			tokenized_verse = [w for w in tokenized_verse if "!" not in w]
			tokenized_verse = [w for w in tokenized_verse if "-" not in w]
			tokenized_verse = [w for w in tokenized_verse if "+" not in w]
			tokenized_verse = [w for w in tokenized_verse if "[" not in w]
			tokenized_verse = [w for w in tokenized_verse if "]" not in w]
			tokenized_verse = [w for w in tokenized_verse if "----------------------------------------" not in w]
			tokenized_verse = [w for w in tokenized_verse if "-----------------------------" not in w]
			tokenized_verse = [w for w in tokenized_verse if "////////////////////" not in w]
			tokenized_verse = [w for w in tokenized_verse if not re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3})', w)]
			tokenized_verse = [w for w in tokenized_verse if not re.match('(\d+.\d+ | \d+)euro',w) or not re.match('(\d+.\d+ | \d+)€',w)]
			tokenized_verse = [w for w in tokenized_verse if not w.isdigit()]
			#tokenized_verse = [w for w in tokenized_verse if w.isdigit() != True]
			# remove remaining tokens that are not alphabetic
			#tokenized_verse = [w for w in tokenized_verse if w.isalpha()]
			"""
			tk = []
			for word in tokenized_verse:
				if word.isdigit():
					tk.append(word)
				else :
					t = ''.join([w for w in word if not w.isdigit()])
					tk.append(t)
			#
			tokenized_verse = tk
			"""
			#
			words.extend(tokenized_verse)

		return words
	"""##################################################################"""
	def extractWordsFromData(self, data):
		words = []
		for line in data:
			# tokenized_verse = nl.word_tokenize(ll, language='italian')
			# tokenized_verse_l = re.split(" ", line)
			# tokenized_verse = [w.lower() for w in tokenized_verse if "\"" not in w]
			# remove puntaction
			# tt = []
			# for l in line :
			# tt.append(l.replace('[^\w\s]', ''))
			tokenized_verse = [w.lower() for w in line if "\"" not in w]
			tokenized_verse = [w for w in tokenized_verse if "," not in w]
			tokenized_verse = [w for w in tokenized_verse if ";" not in w]
			tokenized_verse = [w for w in tokenized_verse if ")" not in w]
			tokenized_verse = [w for w in tokenized_verse if "(" not in w]
			tokenized_verse = [w for w in tokenized_verse if "." not in w]
			tokenized_verse = [w for w in tokenized_verse if "\'" not in w]
			tokenized_verse = [w for w in tokenized_verse if ":" not in w]
			tokenized_verse = [w for w in tokenized_verse if "?" not in w]
			tokenized_verse = [w for w in tokenized_verse if "!" not in w]
			tokenized_verse = [w for w in tokenized_verse if "-" not in w]
			tokenized_verse = [w for w in tokenized_verse if "+" not in w]
			tokenized_verse = [w for w in tokenized_verse if "[" not in w]
			tokenized_verse = [w for w in tokenized_verse if "]" not in w]
			tokenized_verse = [w for w in tokenized_verse if "----------------------------------------" not in w]
			tokenized_verse = [w for w in tokenized_verse if "-----------------------------" not in w]
			tokenized_verse = [w for w in tokenized_verse if "////////////////////" not in w]
			tokenized_verse = [w for w in tokenized_verse if
							   not re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3})', w)]
			tokenized_verse = [w for w in tokenized_verse if
							   not re.match('(\d+.\d+ | \d+)euro', w) or not re.match('(\d+.\d+ | \d+)€', w)]
			tokenized_verse = [w for w in tokenized_verse if not w.isdigit()]
			# tokenized_verse = [w for w in tokenized_verse if w.isdigit() != True]
			# remove remaining tokens that are not alphabetic
			# tokenized_verse = [w for w in tokenized_verse if w.isalpha()]
			"""
			tk = []
			for word in tokenized_verse:
				if word.isdigit():
					tk.append(word)
				else :
					t = ''.join([w for w in word if not w.isdigit()])
					tk.append(t)
			#
			tokenized_verse = tk
			"""
			#
			words.extend(tokenized_verse)

		return words
	"""##################################################################"""

	def extractWordsTicketString(self, data):

		words = []
		tokenized_verse = ""
		i = 0
		try:
			for line in data:
				tokenized_verse = re.split(" ", line)
				tokenized_verse = [w for w in tokenized_verse if "\"" not in w]
				tokenized_verse = [w for w in tokenized_verse if "," not in w]
				tokenized_verse = [w for w in tokenized_verse if ";" not in w]
				tokenized_verse = [w for w in tokenized_verse if ")" not in w]
				tokenized_verse = [w for w in tokenized_verse if "(" not in w]
				tokenized_verse = [w for w in tokenized_verse if "." not in w]
				tokenized_verse = [w for w in tokenized_verse if "\'" not in w]
				tokenized_verse = [w for w in tokenized_verse if ":" not in w]
				tokenized_verse = [w for w in tokenized_verse if "?" not in w]
				tokenized_verse = [w for w in tokenized_verse if "!" not in w]
				tokenized_verse = [w for w in tokenized_verse if "-" not in w]
				tokenized_verse = [w for w in tokenized_verse if "+" not in w]
				tokenized_verse = [w for w in tokenized_verse if "[" not in w]
				tokenized_verse = [w for w in tokenized_verse if "]" not in w]
				tokenized_verse = [w for w in tokenized_verse if "----------------------------------------" not in w]
				tokenized_verse = [w for w in tokenized_verse if "-----------------------------" not in w]
				tokenized_verse = [w for w in tokenized_verse if "////////////////////" not in w]
				tokenized_verse = [w for w in tokenized_verse if
								   not re.match('(0[1-9]|[12]\d|3[01]|[1-9])(\/)(0[1-9]|[1-9]|1[0-2])(\/)([12]\d{3})', w)]
				tokenized_verse = [w for w in tokenized_verse if
								   not re.match('(\d+.\d+ | \d+)euro', w) or not re.match('(\d+.\d+ | \d+)€', w)]
				tokenized_verse = [w for w in tokenized_verse if not w.isdigit()]
				words.extend(tokenized_verse)
				i = i + 1
		except Exception as e:
			print("Exception " + str(e) + "last token inserted " + str(tokenized_verse) + " linea" + str(i))
		return words

	"""##################################################################"""

	def tokenizeTickets(self):
		tokenized_tickets = []
		for ticket in self.data:
			ticket = ticket.replace('\n', ' ').replace('\r', '')
			tokenized_verse = nl.word_tokenize(ticket, language='italian')
			tv = ' '.join(tokenized_verse)
			tokenized_tickets.append(tv)

		self.data = tokenized_tickets
		return tokenized_tickets

	"""##################################################################"""

	def toLower(self):
		tickets_to_lower = []
		for ticket in self.data:
			t = ticket.lower()
			tickets_to_lower.append(t)

		self.data = tickets_to_lower
		return tickets_to_lower

	"""##################################################################"""

	def removeStopWords(self,tickets,targets):
		self.data = tickets
		words = []
		ticketsNoStopWords = []
		targetsNoStopWords = []
		count_line = 0
		for line in self.data:
			targetsNoStopWords.append(targets[count_line])
			tok_line = re.split(" ", line)
			new_tok_line = []
			count_words_in_line = len(tok_line)
			for word in tok_line :
				if word in stopwords.words("italian") :
					count_words_in_line = count_words_in_line - 1
					if count_words_in_line == 0 :
						targetsNoStopWords = targetsNoStopWords[:-1]
					continue
				else :
					new_tok_line.append(word)
			if len(new_tok_line) > 0:
				ticketsNoStopWords.append(new_tok_line)
			count_line = count_line + 1
		self.data = ticketsNoStopWords

		return ticketsNoStopWords,targetsNoStopWords

	"""##################################################################"""
	def removeStopWordsToString(self, tickets, targets):
		self.data = tickets
		words = []
		ticketsNoStopWords = []
		targetsNoStopWords = []
		count_line = 0
		for line in self.data:
			targetsNoStopWords.append(targets[count_line])
			tok_line = re.split(" ", line)
			new_tok_line = ""
			count_words_in_line = len(tok_line)
			for word in tok_line:
				if word in stopwords.words("italian"):
					count_words_in_line = count_words_in_line - 1
					if count_words_in_line == 0:
						targetsNoStopWords = targetsNoStopWords[:-1]
					continue
				else:
					new_tok_line = new_tok_line + " " + word
			if len(new_tok_line) > 0 and new_tok_line != "":
				ticketsNoStopWords.append(new_tok_line.strip())
			count_line = count_line + 1
		self.data = ticketsNoStopWords
		#
		return ticketsNoStopWords, targetsNoStopWords
	"""##################################################################"""

	def removeStopWordsFromTicket(self,ticket):
		words = []
		count_line = 0
		tok_line = re.split(" ", ticket)
		new_tok_line = []
		count_words_in_line = len(tok_line)
		for word in tok_line :
			if word in stopwords.words("italian") :
				count_words_in_line = count_words_in_line - 1
			else :
				new_tok_line.append(word)
		ticketNS = ' '.join(map(str,new_tok_line))
		return ticketNS