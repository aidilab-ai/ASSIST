from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

"""#################################################################"""

def printConfusionMatrix(target,predicted,labels):

	conf_mat = confusion_matrix(target, predicted)
	fig, ax = plt.subplots(figsize=(10,10))
	sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=labels, yticklabels=labels)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show()
