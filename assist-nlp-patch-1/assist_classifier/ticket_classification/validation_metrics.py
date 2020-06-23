from sklearn.metrics import confusion_matrix
import utility as ut
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

class Metrics(object):
	def __init__(self, target, prediction, labels):
		self.target = target
		self.prediction = prediction
		self.labels = labels

	"""How many selected items are relevant?"""
	def getPrecision(self):
		precision = sk.metrics.precision_score(self.target, self.prediction)
		return precision

	"""How many relevant items are selected?"""
	def getRecall(self):
		recall = sk.metrics.recall_score(self.target,self.prediction)
		return recall

	def getF1(self):
		f1 = sk.metrics.recall_score(self.target,self.prediction)
		return f1

	def getConfusionMatrix(self):
		prediction = ut.fromOneHotToTerm(self.labels, self.prediction)
		target = ut.fromOneHotToTerm(self.labels, self.target)
		conf_mat = confusion_matrix(target, prediction, labels=self.labels)
		return conf_mat

	def printConfusionMatrix(self, conf_mat):
		fig, ax = plt.subplots(figsize=(10, 10))
		sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=self.labels, yticklabels=self.labels)
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		plt.show()

	def precisionAtK(self, actual, predicted, k):
		act_set = set(actual)
		pred_set = set(predicted[:k])
		result = len(act_set & pred_set)

		return result

"""##################################################################"""
