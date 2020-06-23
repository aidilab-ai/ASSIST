import warnings
import sklearn.model_selection as sk
import numpy as np
from pathlib import Path
import random
import re
import math
import os

"""##################################################################"""

def deprecated(message):
  def deprecated_decorator(func):
      def deprecated_func(*args, **kwargs):
          warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                        category=DeprecationWarning,
                        stacklevel=2)
          warnings.simplefilter('default', DeprecationWarning)
          return func(*args, **kwargs)
      return deprecated_func
  return deprecated_decorator

"""##################################################################"""

def get_train_and_test(input,target,test_size=0.33):
    X_train, X_test, y_train, y_test = sk.train_test_split(input, target, test_size=test_size, random_state=42, shuffle=True)
    #
    return X_train, X_test, y_train, y_test

def get_train_testWithSequenceFeatures(input, target, featureSeq, test_size=0.33):
	# To shuffle sequences
	ids = random.sample(range(len(input)), len(input))

	#percentage of test size :
	ts_percentage = test_size * 100
	test_dim = len(input) / 100 * ts_percentage
	training_dim = len(input) - int(test_dim)
	#
	#training ids
	training_ids = ids[0:training_dim]
	test_ids = ids[training_dim:len(input)]

	# Generator for batch
	train_input, train_target, train_feature = [], [], []
	test_input, test_target, test_feature = [], [], []

	#
	for id in training_ids:
		train_input.append(input[id])
		train_target.append(target[id])
		train_feature.append(featureSeq[id])
	#
	for id in test_ids:
		test_input.append(input[id])
		test_target.append(target[id])
		test_feature.append(featureSeq[id])

	return train_input, train_target, train_feature, test_input, test_target, test_feature

"""##################################################################"""
def __balanceSubSamples(config,ticket, target, labels, max_number = 3000):
    class_xs = []
    min_elems = None

    counting_elem = 0
    for yi in labels:
        count = 0

        for t in target :
            if count < max_number and (t == yi) :
                count = count + 1
                element = ticket[counting_elem]
                writeInFile(element, "tickets_balanced", config)
                writeInFile(yi,"target_balanced",config)
            counting_elem = counting_elem + 1
            if counting_elem >= len(ticket) :
                counting_elem = 0
    return None

def balanceSubSamples(config,tickets, target, labels, max_number = 3000):
	class_xs = []
	min_elems = None
	tickets_targets =  []
	print("Tickets len " + str(len(tickets)))
	print("Targets len " + str(len(target)))
	for i, ticket in enumerate(tickets):
		if ticket != " " and ticket != None:
			tickets_targets.append(ticket + " xXxXx " +target[i])

	random.shuffle(tickets_targets)
	counting_elem = 0
	for yi in labels:
		count = 0
		for ticket_target in tickets_targets :
			tt_split = re.split(" xXxXx ", ticket_target)
			target = tt_split[1]
			ticket = tt_split[0]
			if count < max_number and (target.strip() == yi.strip()) :
				count = count + 1
				#element = ticket[counting_elem]
				writeInFile(ticket.rstrip(), "tickets_balanced_"+str(max_number), config)
				writeInFile(target.rstrip(),"target_balanced_"+str(max_number),config)
				counting_elem = counting_elem + 1
				if counting_elem >= len(ticket) :
					counting_elem = 0
	return None

"""##################################################################"""

def balanceSubSamplesReturnArray(tickets, target, labels, max_number = 3000):
	new_tickets = []
	new_targets = []
	class_xs = []
	min_elems = None
	tickets_targets =  []
	for i, ticket in enumerate(tickets):
		if ticket != " " and ticket != None:
			tickets_targets.append(ticket + " xXxXx " +target[i])

	random.shuffle(tickets_targets)
	counting_elem = 0
	for yi in labels:
		count = 0
		for ticket_target in tickets_targets :
			tt_split = re.split(" xXxXx ", ticket_target)
			target = tt_split[1]
			ticket = tt_split[0]
			if count < max_number and (target.strip() == yi.strip()) :
				count = count + 1
				#element = ticket[counting_elem]
				new_tickets.append(ticket)
				new_targets.append(target)
				counting_elem = counting_elem + 1
				if counting_elem >= len(ticket) :
					counting_elem = 0

	return new_tickets,new_targets

"""##################################################################"""

def writeInFile(data,data_path,config):
    my_file = Path(config.main_path + '/splittedData/'+ data_path + '.txt')
    if my_file.is_file():
        textfile = open(config.main_path + '/splittedData/'+ data_path + '.txt', 'a', encoding=config.csv_encoding)
        textfile.write("%s\n" % data)
        textfile.close()
    else:
        textfile = open(config.main_path + '/splittedData/'+ data_path + '.txt', 'w+', encoding=config.csv_encoding)
        textfile.write("%s\n" % data)
        textfile.close()

"""##################################################################"""

def fromOneHotToTerm(mapping, vectors):
	thereshold = -1
	dict_map = dict()
	dict_map.update([(i, m) for i, m in enumerate(mapping)])
	m = []
	for vector in vectors:
		el = np.max(vector)
		if el >= thereshold :
			n_el = np.where(vector==el)
			if len(n_el[0]) > 0 :
				mapped = dict_map[n_el[0][0]]
			else:
				mapped = "Not found"
		else :
			mapped = "Not found"
		m.append(mapped)
	return m

def fromOneHotToTermWithSorting(mapping, vectors):
    thereshold = -1
    dict_map = dict()
    dict_map.update([(i, m) for i, m in enumerate(mapping)])
    m = []
    pr = []
    for vector in vectors:
        sort= np.argsort(vector)
        max_index_val = sort[len(sort) - 1]
        second_index_max_val = sort[len(sort) - 2]
        max_val = vector[max_index_val]
        second_max_val = vector[second_index_max_val]
        n_el = np.where(vector==max_val)
        if len(n_el[0]) > 0 :
        	mapped = dict_map[n_el[0][0]]
        else:
        	mapped = "Not found"

        nn_el = np.where(vector==second_max_val)
        if len(nn_el[0]) > 0 :
        	mappedSecond = dict_map[nn_el[0][0]]
        else:
        	mappedSecond = "Not found"

        top_difference = max_val - second_max_val
        #
        #Entropy
        entropy = Entropy(vector)
        p = "Top : " + str(mapped) + " | Second : " + str(mappedSecond) + " | Distance : " + str(top_difference) + " | Entropy " + str(entropy)

        m.append(mapped)
        pr.append(p)
    return m, pr, mapped, mappedSecond


"""##################################################################"""

def fromOneHotToArrayOfTerms(mapping, vectors):
	thereshold = -1
	dict_map = dict()
	dict_map.update([(i, m) for i, m in enumerate(mapping)])
	m = []
	pr = []
	for vector in vectors:
		sort = np.argsort(vector)
		values = []
		for i in range(len(sort)-1,0,-1):
			vv = vector[sort[i]]
			values.append(vv)
			pr.append(values)

		mapped_array = []
		for j in range(len(values)):
			n_el =  np.where(vector==values[j])
			mapped = None
			if len(n_el[0]) > 0 :
				mapped = dict_map[n_el[0][0]]
			else:
				mapped = "Not found"
			mapped_array.append(mapped)
		m.append(mapped_array)

	return m,pr

"""##################################################################"""

def Entropy(array_denProb):
	num_base = len(array_denProb)
	inner_entropy = []
	for i in range(len(array_denProb)):
		inner_entropy.append(array_denProb[i]*math.log(array_denProb[i],num_base))

	entropy = - sum(inner_entropy)

	return entropy


"""##################################################################"""

def removeIdenticalTickets(tickets,targets):
	data_2 = tickets
	olready_checked = []
	counter = 0
	print("tickets " + str(len(tickets)) + " targets " + str(len(targets)))
	for i in range(len(tickets)):
		d2verify = tickets[i]
		target2v = targets[i]
		if d2verify not in olready_checked :
			for j in range(len(data_2)):
				k = i+j
				if k < len(data_2) and k != i:
					data2_2verify = data_2[k]
					if k < len(targets):
						target2_2v = targets[k]
					if d2verify == data2_2verify and target2v != target2_2v:
						print("Ticket identici:  " + str(data2_2verify) + " | id : " + str(i) + " con target " + target2v + " ed id " + str(i+j) + " con target " + target2_2v)
						olready_checked.append(d2verify)
						counter = counter + 1
	new_tickets = []
	new_targets = []
	for h in range(len(tickets)):
		if tickets[h] not in olready_checked:
			new_tickets.append(tickets[h])
			new_targets.append(targets[h])

	return new_tickets, new_targets

"""##################################################################"""

def removeIdenticalTicketsFromNew(tickets,targets, start, end):
	data_2 = tickets
	olready_checked = []
	counter = 0
	for i in range(start,end):
		d2verify = tickets[i]
		target2v = targets[i]
		if d2verify not in olready_checked :
			for j in range(len(data_2)):
				k = i+j
				if k < len(data_2) and k != i:
					data2_2verify = data_2[k]
					if k < len(targets):
						target2_2v = targets[k]
					if d2verify == data2_2verify and target2v != target2_2v:
						print("Ticket identici:  " + str(data2_2verify) + " | id : " + str(i) + " con target " + target2v + " ed id " + str(i+j) + " con target " + target2_2v)
						olready_checked.append(d2verify)
						counter = counter + 1
	new_tickets = []
	new_targets = []
	for h in range(len(tickets)):
		if tickets[h] not in olready_checked:
			new_tickets.append(tickets[h])
			new_targets.append(targets[h])

	return new_tickets, new_targets

"""##################################################################"""

def renameClassLabel(targets,labels,old_name,new_name):
	for i in range(len(targets)):
		targets[targets.index(old_name,i)] = new_name

	labels[labels.index(old_name)] = new_name

"""##################################################################"""

def checkDircetoryExistence(directory_path):
	if os.path.exists(directory_path):
		return True
	else:
		return False

"""##################################################################"""

def createDirectory(directory_path):
	os.makedirs(directory_path, exist_ok=True)

