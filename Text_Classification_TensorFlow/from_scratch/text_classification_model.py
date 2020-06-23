from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.train import Saver
from gensim.models import Word2Vec
import tensorflow.compat.v1 as tf
import numpy as np
import os

class TextClassifier:

    def __init__(self, training, vocabulary_size, embedding_size, lstm_units, output_units, word2vec_model_path=None):

        self.lstm_units = lstm_units

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):

            # Placeholders for inputs and target labels
            self.inputs = tf.placeholder(tf.int32, shape=[None, None]) # [batch_size, max_sequence_length]
            self.labels = tf.placeholder(tf.int32, shape=[None, None]) # [batch_size, output_units]

        with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):

            if word2vec_model_path:

                # Load pre-trained word embeddings
                model = Word2Vec.load(word2vec_model_path)
                pretrained_embeddings = model.wv.syn0
                unk_embedding = np.random.normal(size=pretrained_embeddings.shape[1]).astype(np.float32)
                pad_embedding = np.random.normal(size=pretrained_embeddings.shape[1]).astype(np.float32)
                all_embeddings = np.vstack([pretrained_embeddings, unk_embedding, pad_embedding])
                embeddings_initializer = all_embeddings
                trainable_embeddings = False
                del model

            else:

                # Trainable word embeddings
                embeddings_initializer = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
                trainable_embeddings = True

            embeddings = tf.get_variable(name='embeddings', initializer=embeddings_initializer, trainable=trainable_embeddings)

            # This is fundamentally equal to embeddings[train_inputs] (numpy indexing). However, the
            # advantage of using 'embedding_lookup' is that such computation is done in parallel
            retrieved_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs) # [batch_size, max_sequence_length, embedding_size]

        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):

            # Recurrent layer
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units=lstm_units)
            outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, inputs=retrieved_embeddings, dtype=tf.float32) # 'final_states' has shape [batch_size, lstm_units]
            final_states_c, final_states_h = final_states.c, final_states.h

            if training:

                with tf.variable_scope('dropout', reuse=tf.AUTO_REUSE):

                    # Dropout layer
                    self.keep_prob = tf.placeholder(tf.float32)
                    final_states_c = tf.nn.dropout(final_states_c, keep_prob=self.keep_prob)

        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):

            # Model output
            self.logits = tf.layers.dense(final_states_c, output_units, activation=tf.nn.relu) # [batch_size, output_units]

            # Compute accuracy (relative to the current batch)
            correct_predictions = tf.equal( tf.argmax(self.logits, 1), tf.argmax(self.labels, 1) )
            self.accuracy = tf.reduce_mean( tf.cast(correct_predictions, tf.float32) )

            # Loss function to minimize
            # self.class_weights = tf.placeholder(tf.float32, shape=[1, None])
            # samples_weights = tf.reduce_sum(self.class_weights * tf.cast(self.labels, tf.float32), axis=1)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            # weighted_cross_entropy = cross_entropy * samples_weights

            self.loss = tf.reduce_mean(cross_entropy)

            if training:

                with tf.variable_scope('regularization', reuse=tf.AUTO_REUSE):

                    # Add L2 regularization
                    self.l2_regularization = tf.placeholder(tf.float32)
                    l2_losses = [tf.nn.l2_loss(var) for var in tf.trainable_variables() if not 'bias' in var.name]
                    self.loss += self.l2_regularization * tf.reduce_sum(l2_losses)


        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):

            if training:
                # Define the SGD optimizer to minimize the loss function
                self.learning_rate = tf.placeholder(tf.float32)
                self.training_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope('summaries', reuse=tf.AUTO_REUSE):

            # Add some summaries for TensorBoard
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)


    def _generate_batches(self, input_sequences, labels, batch_size, shuffle=True):

        if shuffle:
            perm = np.random.permutation( len(input_sequences) )
        else:
            perm = np.arange( len(input_sequences) )

        index = 0
        while index < len(input_sequences):

            indexes = perm[index:index+batch_size]
            yield input_sequences[indexes], labels[indexes]
            index += batch_size


    def _make_hparams_string(self, lstm_units, batch_size, dropout_keep_prob, learning_rate, l2_regularization):
        return f'lu={lstm_units}_bs={batch_size}_kp={dropout_keep_prob}_lr={learning_rate}_l2reg={l2_regularization}'


    def fit(self, input_sequences, labels, batch_size, epochs, validation_fraction=0.0, dropout_keep_prob=0.7, learning_rate=0.01, l2_regularization=1e-5, checkpoint_dir=None, tensorboard_logdir='/tmp/tensorboard'):

        # Train the model
        with tf.Session() as session:

            session.run( tf.global_variables_initializer() )

            # Create a TensorBoard file writer
            hparams_string = self._make_hparams_string(self.lstm_units, batch_size, dropout_keep_prob, learning_rate, l2_regularization)
            writer_training = tf.summary.FileWriter( os.path.join(tensorboard_logdir, 'training_{}'.format(hparams_string)))
            writer_validation = tf.summary.FileWriter( os.path.join(tensorboard_logdir, 'validation_{}'.format(hparams_string)))
            #writer_training.add_graph(session.graph)

            # Merge all summaries into one tensor
            merged_summary = tf.summary.merge_all()

            if validation_fraction > 0:
                x_train, x_validation, y_train, y_validation = train_test_split(input_sequences, labels, test_size=validation_fraction)
            else:
                x_train, y_train = input_sequences, labels

            print('Training started ({})...'.format(hparams_string))
            global_step = 0
            # class_weights = np.sum(labels, axis=0).reshape(1, -1) / len(labels)
            for epoch_number in range(epochs):

                for inputs_batch, labels_batch in self._generate_batches(x_train, y_train, batch_size, shuffle=True):

                    feed_dict = {
                        self.inputs: inputs_batch,
                        self.labels: labels_batch,
                        self.keep_prob: dropout_keep_prob,
                        self.learning_rate: learning_rate,
                        self.l2_regularization: l2_regularization
                        # self.class_weights: class_weights
                    }
                    _, cur_loss, batch_acc, logits, summary = session.run([self.training_step,
                                                                           self.loss,
                                                                           self.accuracy,
                                                                           self.logits,
                                                                           merged_summary], feed_dict=feed_dict)

                    # Record values to be later visualized using TensorBoard
                    writer_training.add_summary(summary, global_step=global_step)

                    if validation_fraction > 0:

                        feed_dict = {
                            self.inputs: x_validation,
                            self.labels: y_validation,
                            self.keep_prob: 1.0,
                            self.l2_regularization: 0.0
                            # self.class_weights: class_weights
                        }
                        summary = session.run(merged_summary, feed_dict=feed_dict)

                        # Write summaries for validation
                        writer_validation.add_summary(summary, global_step=global_step)

                    global_step += 1

                # Compute training accuracy after an epoch has passed
                feed_dict = {
                    self.inputs: x_train,
                    self.labels: y_train,
                    self.keep_prob: 1.0,
                    self.l2_regularization: 0.0
                }
                epoch_train_accuracy = session.run(self.accuracy, feed_dict=feed_dict)

                print(f'Finished epoch {epoch_number}. Achieved accuracy:')
                print(f'    Training: {epoch_train_accuracy * 100:.2f}%')

                if validation_fraction > 0:

                    # Compute validation accuracy after an epoch has passed
                    feed_dict = {
                        self.inputs: x_validation,
                        self.labels: y_validation,
                        self.keep_prob: 1.0,
                        self.l2_regularization: 0.0
                    }
                    epoch_validation_accuracy = session.run(self.accuracy, feed_dict=feed_dict)
                    print(f'    Validation: {epoch_validation_accuracy * 100:.2f}%')

            print(f'Training ended ({hparams_string})...')

            if checkpoint_dir:

                # Save a checkpoint of the model
                saver = Saver()
                saver.save(session, os.path.join(checkpoint_dir, 'checkpoint.ckpt'))


    def restore(self, checkpoint_dir, recurrent_part_only=False):

        session = tf.Session()

        if recurrent_part_only:
            var_list = tf.trainable_variables(scope='lstm')
        else:
            var_list = None

        saver = Saver(var_list=var_list)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(session, checkpoint)

        return session


    def predict(self, session, input_sequences, labels, batch_size):

        # Test the model
        # with tf.Session() as session:

        #self._restore(session, checkpoint_dir)

        logits = []
        for inputs_batch, labels_batch in self._generate_batches(input_sequences, labels, batch_size, shuffle=False):

            feed_dict = {
                self.inputs: inputs_batch,
                self.labels: labels_batch
            }

            logits_batch = session.run(self.logits, feed_dict=feed_dict)
            logits.append(logits_batch)

        logits = np.concatenate(logits)
        return logits
