import sys
sys.path.insert(0, './assist_classifier/ticket_classification')

import pandas as pd
import numpy as np

import filter_data
import vocabulary
import tokenizer
import config_in
import skipgram
import model
import data


config_directory = "/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/assist_classifier/ticket_classification/config/fcariaggi/"
config_category_file = "category_config.json"
config_priority_file = "priority_config.json"
config_connector_file = "connector_config.json"

# Read configuration file for category classification
configModelClass = config_in.Config()
configModelClass.configFromFile(config_directory + config_category_file)
configModelClass.is_test = True
labelsClass = configModelClass.labels
dataMC = data.Data(configModelClass)

# Read configuration file for priority classification
configModelPriority = config_in.Config()
configModelPriority.configFromFile(config_directory + config_priority_file)
configModelPriority.is_test = True
labelsPriority = configModelPriority.labels
#dataMC = dt.Data(configModelPriority)

# Load Existing Vocabulary
voc = vocabulary.Vocabulary(configModelClass)
dictionary = voc.loadDictionary("vocabulary")
reverse_dict = voc.getReverseDictionary(dictionary)

if configModelClass.use_pretrained_embs:
	skip = skipgram.SkipgramModel(configModelClass)
	skipgramModel = skip.get_skipgram()
	voc = vocabulary.Vocabulary(configModelClass)
	reverse_dict = voc.getReverseDictionary(dictionary)
	skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
	configModelClass.skipgramEmbedding = skipgramEmbedding

	voc_priority = vocabulary.Vocabulary(configModelPriority)
	dictionary_priority = voc_priority.loadDictionary("vocabulary")
	reverse_dict_priority = voc.getReverseDictionary(dictionary_priority)
	skipgramEmbedding_p = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict_priority)
	configModelPriority.skipgramEmbedding = skipgramEmbedding_p


def cleanData(data):
    tok = tokenizer.Tokenizer(data)
    tok.tokenizeTickets()
    tickets_to_lower = tok.toLower()
    tickets_no_sw = tok.removeStopWordsFromTicket(tickets_to_lower[0])
    return tickets_no_sw

# ticket = [str(response['description'])]
# tickets = [["adsl assente telefono non funziona"],
#            ["inviata mail al consulente marco parigi e a riccardo signorini per la gestione del cliente : il cliente è senza tlc , per cui occorre collegare le linee indicate su mail ad abilis senza np . riscontrato problema per il collegamento della linea rtg di altra azienda 0543720306 , per cui occorre l'installazione di un apparato aggiuntivo ( patton ) . da verificare se occorre fare integrazione contrattuale o meno . a chiusura potete riassociare ticket ad attivazioni ."],
#            ["il cli mi dice che per connettersi al server non si connette piu con l'indirizzo ip pubblico , mentre con la vpn si"]]

x_test_path = '/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/data/models_and_data/fcariaggi/models/category/model/data/tickets_test_QIT.txt'
y_test_path = '/home/anfri/Lavoro-AiDiLab/ASSIST/assist-modified/data/models_and_data/fcariaggi/models/category/model/data/targets_test_QIT.txt'
max_rows = 2500

tickets = np.genfromtxt(x_test_path, dtype=str, delimiter='\n', max_rows=max_rows).reshape((-1, 1))
category_labels = np.genfromtxt(y_test_path, dtype=str, delimiter='\n', max_rows=max_rows)

print(tickets.shape)
print(category_labels.shape)

categories_mapping = {
    0: 'Thrash',
    1: 'Amministrativa',
    2: 'Tecnica',
    3: 'Informazioni',
    4: 'Marketing',
    5: 'Altro'
}

descriptions, sequences, categoryIDS, priorityIDS = [], [], [], []
for ticket in tickets:

    ticket_cleaned = cleanData(ticket)
    ticket_array = ticket_cleaned.split(" ");

    # Create Sequences and HotVectors for the Target
    tickets_sequences = dataMC.createDataSequence([ticket_array], dictionary)
    ticket_sequence = tickets_sequences[0]

    descriptions.append(ticket)
    sequences.append(ticket_sequence)

    # Trashing ticket with too much words out of vocabulary
    filtdata = filter_data.FilterData(configModelClass, labelsClass)
    trashIT = filtdata.trashingTicket(ticket_sequence, dictionary)

    if trashIT:

        params = {"categoryId": "Trash", "priorityId": 0}

        # Invio il ticket con la label: Cestino
        categoryIDS.append("Trash")
        priorityIDS.append(1)

    else:
        categoryIDS.append(0)
        priorityIDS.append(1)


# Aggrego i tickets
tickets = [{"description": desc, "sequence": seq, "categoryID": cid, "priorityID": pid}
            for desc, seq, cid, pid in zip(descriptions, sequences, categoryIDS, priorityIDS)]


print('Predicting categories...')
# Carico il Modello delle categorie
out_array, class_predicted = model.runPredictionTickets(configModelClass, tickets, labelsClass)

for i in range(len(tickets)):

    if tickets[i]['categoryID'] != "Trash":
        tickets[i]['categoryID'] = configModelClass.labels_map.get(class_predicted[i])
    else:
        tickets[i]['categoryID'] = 0


# print('Predicting priorities...')
# # Carico il Modello delle priorità
# out_array, class_predicted = model.runPredictionTickets(configModelPriority, tickets, labelsPriority)
#
# # Aggiorno l'oggetto
# for i in range(len(tickets)):
#     if tickets[i]['categoryID'] != 0:
#         tickets[i]['priorityID'] = labelsPriority.index(class_predicted[i]) + 1

# Evaluating model accuracy
y_true = category_labels
y_pred = np.array( list( map(lambda t: categories_mapping[t['categoryID']], tickets) ) )

matchings = y_pred == y_true
accuracy = np.mean(matchings) * 100

print('Accuracy: {:.2f}%'.format(accuracy))

# print('Predicted categories:')
# for ticket, category in zip(tickets, category_labels):
#     print('  {} (true category: {})'.format(categories_mapping[ticket['categoryID']], category))

# print('Predicted priorities:')
# for ticket in tickets:
#     print('  {}'.format(ticket['priorityID']))
