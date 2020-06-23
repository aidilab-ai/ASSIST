import tensorflow.compat.v1 as tf
import numpy as np

vocabulary_size = 13046
embedding_size = 256
num_noise = 1
learning_rate = 1e-3
batch_size = 1024
epochs = 10

def make_hparam_string(embedding_size, num_noise, learning_rate, batch_size, epochs):
    return f'es={embedding_size}_nn={num_noise}_lr={learning_rate}_bs={batch_size}_e={epochs}'

# These are the hidden layer weights
embeddings = tf.get_variable(name='embeddings', initializer=tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), trainable=True)

# 'nce' stands for 'Noise-contrastive estimation' and represents a particular loss function.
# Check https://www.tensorflow.org/tutorials/representation/word2vec for more details.
# 'nce_weights' and 'nce_biases' are simply the output weights and biases.
# NOTE: for some reason, even though output weights will have shape (embedding_size, vocabulary_size),
#       we have to initialize them with the shape (vocabulary_size, embedding_size)
nce_weights = tf.get_variable(name='output_weights',
                              initializer=tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)),
                              trainable=True)
nce_biases = tf.get_variable(name='output_biases', initializer=tf.constant_initializer(0.1), shape=[vocabulary_size], trainable=True)

# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[None])    # [batch_size]
train_labels = tf.placeholder(tf.int32, shape=[None, 1]) # [batch_size, 1]

# This allows us to quickly retrieve the corresponding word embeddings for each word in 'train_inputs'
matched_embeddings = tf.nn.embedding_lookup(embeddings, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=train_labels,
                                     inputs=matched_embeddings,
                                     num_sampled=num_noise,
                                     num_classes=vocabulary_size))

# Use the SGD optimizer to minimize the loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Add some summaries for TensorBoard
loss_summary = tf.summary.scalar('nce_loss', loss)
input_embeddings_summary = tf.summary.histogram('input_embeddings', embeddings)
output_embeddings_summary = tf.summary.histogram('output_embeddings', nce_weights)

################################################################################

# Load data
target_words = np.genfromtxt('target_words.txt', dtype=int, delimiter='\n').reshape((-1, 1))
context_words = np.genfromtxt('context_words.txt', dtype=int, delimiter='\n').reshape((-1, 1))

# Convert to tensors
target_words_tensor = tf.convert_to_tensor(target_words)
context_words_tensor = tf.convert_to_tensor(context_words)

# Create a tf.data.Dataset object representing our dataset
dataset = tf.data.Dataset.from_tensor_slices((target_words_tensor, context_words_tensor))
dataset = dataset.shuffle(buffer_size=target_words.shape[0])
dataset = dataset.batch(batch_size)

# Create an iterator to iterate over the dataset
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

# Train the model
with tf.Session() as session:

    # Initialize variables
    session.run( tf.global_variables_initializer() )

    merged_summary = tf.summary.merge_all()

    # File writer for TensorBoard
    hparam_string = make_hparam_string(embedding_size, num_noise, learning_rate, batch_size, epochs)
    loss_writer = tf.summary.FileWriter(f'./tensorboard/{hparam_string}')

    global_step = 0
    for epoch in range(epochs):

        session.run(iterator.initializer)
        while True:
            try:
                inputs, labels = session.run(next_batch)

                feed_dict = {train_inputs: inputs[:, 0], train_labels: labels}
                _, cur_loss, all_summaries = session.run([optimizer, loss, merged_summary], feed_dict=feed_dict)

                # Write sumaries to disk
                loss_writer.add_summary(all_summaries, global_step=global_step)
                global_step += 1

                print(f'Current loss: {cur_loss}')

            except tf.errors.OutOfRangeError:
                print(f'Finished epoch {epoch}.')
                break
