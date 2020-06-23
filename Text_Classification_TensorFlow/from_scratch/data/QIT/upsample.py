import pandas as pd
import numpy as np

x = pd.read_csv('x_QIT.txt', delimiter='\n', header=None, dtype=str, names=['Ticket']) #np.genfromtxt('x_QIT.txt', dtype=int, delimiter='\n')
y = pd.read_csv('y_QIT.txt', delimiter='\n', header=None, dtype=str, names=['Label']) #np.genfromtxt('y_QIT.txt', dtype=int, delimiter='\n')

# Count occurences for each label
label_counts = pd.value_counts(y['Label'])
label_counts = zip(label_counts.index, label_counts.values)

# Cut out samples whose class is too much infrequent
min_label_frequency = 100
frequent_labels = [label for (label, count) in label_counts if count >= min_label_frequency]
frequent_labels_mask = y['Label'].apply(lambda label: label in frequent_labels)
x_frequent = x[frequent_labels_mask]
y_frequent = y[frequent_labels_mask]


# Perform upsampling of the less frequent classes
max_label_count = pd.value_counts(y_frequent['Label']).max()

for freq_label in frequent_labels:

    mask = (y_frequent['Label'] == freq_label)

    extracted_labels = y_frequent[mask]
    replications = int( np.ceil( max_label_count / len(extracted_labels) ) ) - 1

    extracted_samples = x_frequent[mask]
    for _ in range(replications):

        x_frequent = pd.concat([x_frequent, extracted_samples])
        y_frequent = pd.concat([y_frequent, extracted_labels])


print( pd.value_counts(y_frequent['Label']) )

x_frequent['Ticket'].to_csv('upsampled/x_QIT.txt', header=False, index=False)
y_frequent['Label'].to_csv('upsampled/y_QIT.txt', header=False, index=False)
