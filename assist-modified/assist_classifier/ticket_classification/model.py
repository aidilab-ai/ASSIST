import tensorflow as tf
import batcher as bc
import utility as ut
import data as dt
import skipgram as sk
import vocabulary as vc
import validation_metrics as vm
import numpy as np
from pathlib import Path
from tensorboard import summary as summary_lib
import platform


"""#############################################################################################################"""
"""
	Encoder Class
"""
class EncoderBidirectional(object):
	#
	def __init__(self,sequences,config):
		self.sequences = sequences
		#
		with tf.name_scope("Embeddings"):
			if config.use_pretrained_embs:
				#self.embeddings = tf.get_variable("InputEmbeddings",word_embeddings, trainable=False)
				Shared_Embedding = tf.get_variable("InputEmbeddings",[len(config.skipgramEmbedding), config.input_word_emb_size],trainable=False)
				self.embeddings = Shared_Embedding.assign(tf.to_float(config.skipgramEmbedding))
				config.skipgramEmbedding  = []
			else:
				self.embeddings = tf.get_variable("InputEmbeddings",[config.vocab_size,config.input_word_emb_size],dtype=tf.float32)
			#

			self.wes = tf.nn.embedding_lookup(self.embeddings, self.sequences) # batch_size x sentence_max_len x word_emb_size
			#
			with tf.name_scope("Encoding"):
				with tf.variable_scope("fw"):
					self.seq_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)
				#
				with tf.variable_scope("bw"):
					self.seq_rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)
			#
			outputs, (fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.seq_rnn_cell_fw,cell_bw=self.seq_rnn_cell_bw,inputs=self.wes,dtype=tf.float32)
			#outputs = tf.contrib.static_bidirectional_rnn(self.seq_rnn_cell_fw, self.seq_rnn_cell_bw, self.wes, dtype=tf.float32)
			# outputs has 2 elements  with dimension batch_size x seq_max_len x encoder_rnn_size
			# (fw_state,bw_state) every item in tuple has dimension batch_size x encoder_rnn_size

			#
			self.encodings = tf.concat((fw_state.h,bw_state.h),axis=1) # final state of bidirectional RNN
			#
			out_fw, out_bw = outputs
			output = tf.concat([out_fw, out_bw], axis=1)
			self.output = tf.transpose(output, [1, 0, 2])

"""#############################################################################################################"""
class BidirectionalStaticEncoder(object):
	#
	def __init__(self, sequences, config):
		self.sequences = sequences
		#
		with tf.name_scope("Embeddings"):
			if config.use_pretrained_embs:
				# self.embeddings = tf.get_variable("InputEmbeddings",word_embeddings, trainable=False)
				Shared_Embedding = tf.get_variable("InputEmbeddings",
												   [len(config.skipgramEmbedding), config.input_word_emb_size],
												   trainable=False)
				self.embeddings = Shared_Embedding.assign(tf.to_float(config.skipgramEmbedding))
				config.skipgramEmbedding = []
			else:
				self.embeddings = tf.get_variable("InputEmbeddings", [config.vocab_size, config.input_word_emb_size],
												  dtype=tf.float32)
			#

			self.wes = tf.nn.embedding_lookup(self.embeddings,
											  self.sequences)  # batch_size x sentence_max_len x word_emb_size
			#
			with tf.name_scope("Encoding"):
				with tf.variable_scope("fw"):
					self.seq_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)
				#
				with tf.variable_scope("bw"):
					self.seq_rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)
			#
			self.outputs = tf.contrib.static_bidirectional_rnn(self.seq_rnn_cell_fw, self.seq_rnn_cell_bw, self.wes, dtype=tf.float32)


"""#############################################################################################################"""
class Encoder(object):
	#
	def __init__(self,sequences,config):
		self.sequences = sequences
		#
		with tf.name_scope("Embeddings"):
			if config.use_pretrained_embs:
				#self.embeddings = tf.to_float(config.skipgramEmbedding)
				Shared_Embedding = tf.get_variable("InputEmbeddings",[len(config.skipgramEmbedding),config.input_word_emb_size], trainable=False)
				self.embeddings = Shared_Embedding.assign(tf.to_float(config.skipgramEmbedding))
				config.skipgramEmbedding = []
			else:
				self.embeddings = tf.get_variable("InputEmbeddings",[config.vocab_size,config.input_word_emb_size],dtype=tf.float32)
			#
			if config.use_embedding_dropout :
				with tf.name_scope("Embedding_dropout"):
					self.embeddings = tf.convert_to_tensor(self.embeddings)
					self.embeddings = tf.nn.dropout(self.embeddings, keep_prob=config.embedding_dropout_keep_prob)

			self.wes = tf.nn.embedding_lookup(self.embeddings,self.sequences) # batch_size x sentence_max_len x word_emb_size
			#
			with tf.name_scope("Encoding"):
				with tf.variable_scope("fw"):
					self.seq_rnn_cell_fw = tf.contrib.rnn.LSTMCell(config.encoder_rnn_size)
					#self.seq_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)

			#
			outputs, final_states = tf.nn.dynamic_rnn(self.seq_rnn_cell_fw,inputs=self.wes,dtype=tf.float32)
			# outputs has 2 elements  with dimension batch_size x seq_max_len x encoder_rnn_size
			# (fw_state,bw_state) every item in tuple has dimension batch_size x encoder_rnn_size

			#
			self.encodings = final_states.h # final state of bidirectional RNN

"""#############################################################################################################"""
class EncoderWithNoEmbedding(object):
	#
	def __init__(self,sequences,config):
		self.sequences = sequences

		with tf.name_scope("Encoding_NoEmbedding"):
			with tf.variable_scope("fw"):
				self.seq_rnn_cell_fw = tf.contrib.rnn.LSTMCell(config.encoder_rnn_size)
				#self.seq_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(config.encoder_rnn_size)

			#
			with tf.variable_scope("dynamic_rnn_no_embedding"):
				outputs, final_states = tf.nn.dynamic_rnn(self.seq_rnn_cell_fw,inputs=tf.to_float(self.sequences),dtype=tf.float32)
			# outputs has 2 elements  with dimension batch_size x seq_max_len x encoder_rnn_size
			# (fw_state,bw_state) every item in tuple has dimension batch_size x encoder_rnn_size

			#
			self.encodings = final_states.h # final state of bidirectional RNN



"""#############################################################################################################"""
"""
	MLP Class
"""

class MLP(object):
	# input shape : batch_size x embedding [input taken from the output of the encoder]
	def __init__(self, input, config, dropout_keep_prob):
		hidden_units = config.hidden_units
		if config.Encoder_type == "LSTM":
			weights = {
				'h_input': tf.Variable(tf.random_normal([config.encoder_rnn_size,hidden_units]), name='h_input'),
				'h_hidden_1' : tf.Variable(tf.random_normal([hidden_units,hidden_units]), name='h_hidden_1'),
				'h_last': tf.Variable(tf.random_normal([hidden_units, config.num_classes]), name='h_last'),
				'h_one': tf.Variable(tf.random_normal([config.encoder_rnn_size, config.num_classes]), name='h_one')
			}
		else:
			weights = {
				'h_input': tf.Variable(tf.random_normal([config.encoder_rnn_size * 2, hidden_units]), name='h_input'),
				'h_hidden_1' : tf.Variable(tf.random_normal([hidden_units,hidden_units]), name='h_hidden_1'),
				'h_last': tf.Variable(tf.random_normal([hidden_units, config.num_classes]), name='h_last'),
				'h_one': tf.Variable(tf.random_normal([config.encoder_rnn_size * 2, config.num_classes]), name='h_one')
			}
		biases = {
			'b_input': tf.Variable(tf.random_normal([hidden_units]), name='b_input'),
			'b_hidden_1': tf.Variable(tf.random_normal([hidden_units]), name='b_hidden_1'),
			'b_last': tf.Variable(tf.random_normal([config.num_classes]), name='b_last')
		}

		self.W_h_input = weights['h_input']
		self.b_input = biases['b_input']
		#
		self.W_h_hidden1 = weights['h_hidden_1']
		self.b_hidden1 = biases['b_hidden_1']
		#
		self.W_h_last = weights['h_last']
		self.b_last = biases['b_last']
		#
		self.W_one_layer = weights['h_one']
		#
		self.input = input
		self.config = config
		self.dropout_keep_prob = dropout_keep_prob

	def multilayer_perceptron_2hidden(self):
		# mul su input x hidden dell'encoder. Shapes : [batch, embedding] x [embedding, hidden_layer] + [bais]
		with tf.name_scope("MLP_layer1"):
			layer_1 = tf.nn.relu(tf.matmul(self.input, self.W_h_input) + self.b_input, name='hidden_layer_1')
			layer_1 = tf.nn.dropout(layer_1, self.dropout_keep_prob)

		with tf.name_scope("MLP_layer2"):
			layer_2 = tf.nn.relu(tf.matmul(layer_1, self.W_h_hidden1) + self.b_hidden1, name='hidden_layer_2')
			layer_2 = tf.nn.dropout(layer_2, self.dropout_keep_prob)

		with tf.name_scope("MLP_layer3"):
			#self.out = tf.nn.sigmoid(tf.matmul(layer_1, self.W_h2) + self.b_2, name='output')
			self.out = tf.add(tf.matmul(layer_2, self.W_h_last), self.b_last, name='output')
			#self.out = tf.nn.softmax(tf.matmul(layer_1, self.W_h2), name='output')
			tf.add_to_collection("output", self.out)
		return self.out, self.W_h_input, self.W_h_hidden1, self.W_h_last

	def multilayer_perceptron(self):
		# mul su input x hidden dell'encoder. Shapes : [batch, embedding] x [embedding, hidden_layer] + [bais]
		with tf.name_scope("MLP_layer1"):
			layer_1 = tf.nn.relu(tf.matmul(self.input, self.W_h_input) + self.b_input, name='hidden_layer_1')
			layer_1 = tf.nn.dropout(layer_1, self.dropout_keep_prob)

		with tf.name_scope("MLP_layer2"):
			#self.out = tf.nn.sigmoid(tf.matmul(layer_1, self.W_h2) + self.b_2, name='output')
			self.out = tf.add(tf.matmul(layer_1, self.W_h_last), self.b_last, name='output')
			#self.out = tf.nn.softmax(tf.matmul(layer_1, self.W_h2), name='output')
			tf.add_to_collection("output", self.out)
		return self.out, self.W_h_input, self.W_h_last

	def linear_layer(self):
		with tf.name_scope("One_layer"):
			#self.out = tf.nn.relu(tf.matmul(self.input, self.W_one_layer) + self.b_last, name='output')
			self.out = tf.add(tf.matmul(self.input, self.W_one_layer), self.b_last, name="output")
			tf.add_to_collection("output",self.out)
		return self.out, self.W_one_layer

"""#############################################################################################################"""
"""
	ticketClassifier
"""

class TicketClassifier(object):
	#
	def __init__(self, config, labels):
		self.x_ph = tf.placeholder(shape=[None, None], dtype=tf.int64, name="x")
		self.y_ph = tf.placeholder(shape=[None, None], dtype=tf.int64, name="y")
		#self.word_embeddings = tf.placeholder(shape=[None, None], dtype=tf.float32, name="word_embeddings")
		self.dropout_keep_prob = tf.placeholder("float", name="dropout_keep_prob")
		self.labels = labels

		with tf.name_scope("Encoder_LSTM"):
			if config.Encoder_type == 'LSTM':
				self.encoder = Encoder(self.x_ph, config)

			elif config.Encoder_type == 'BidirectionalStaticLSTM':
				self.encoder = BidirectionalStaticEncoder(self.x_ph, config)
				self.encoder.encodings = self.encoder.outputs[-1]
			else :
				self.encoder = EncoderBidirectional(self.x_ph, config)

		with tf.name_scope("MLP"):
			self.mlp =  MLP(self.encoder.encodings, config, self.dropout_keep_prob)
			#self.output = self.mlp.one_layer()
			if config.numb_layers == 1:
				self.output, self.W_h_input = self.mlp.linear_layer()
			elif config.numb_layers == 2:
				self.output, self.W_h_input, self.W_h_last = self.mlp.multilayer_perceptron()
			else:
				self.output, self.W_h_input, self.W_h_hidden, self.W_h_last = self.mlp.multilayer_perceptron_2hidden()

		if config.regularization_use:
			with tf.name_scope("Regularization"):
				if config.regularization_type == 'L2':
					if config.numb_layers == 1:
						self.regularizers = tf.nn.l2_loss(self.W_h_input)
					elif config.numb_layers == 2:
						self.regularizers = tf.nn.l2_loss(self.W_h_input) + tf.nn.l2_loss(self.W_h_last)
					else:
						self.regularizers = tf.nn.l2_loss(self.W_h_input) + tf.nn.l2_loss(self.W_h_hidden) + tf.nn.l2_loss(self.W_h_last)

				elif config.regularization_use == 'L1':
					if config.numb_layers == 1:
						self.regularizers = config.regularization_beta * tf.reduce_sum(tf.abs(self.W_h_input))
					elif config.numb_layers == 2:
						self.regularizers = config.regularization_beta * (tf.reduce_sum(tf.abs(self.W_h_input)) + tf.reduce_sum(tf.abs(self.W_h_last)))
					else:
						self.regularizers = config.regularization_beta * (tf.reduce_sum(tf.abs(self.W_h_input)) + tf.reduce_sum(tf.abs(self.W_h_hidden)) + tf.reduce_sum(tf.abs(self.W_h_last)))

		with tf.name_scope("Cost"):
			#softmax = tf.nn.softmax(self.output)
			tf.stop_gradient(self.y_ph)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_float(self.y_ph), logits=self.output)
			self.loss = tf.reduce_mean(cross_entropy)
			# self.loss = tf.losses.log_loss(self.y_ph,self.output, epsilon=0.0001)
			if config.regularization_use:
				self.loss = tf.reduce_mean(self.loss + config.regularization_beta * self.regularizers)

			with tf.device("/cpu:0"):
				self.loss_summary = tf.summary.scalar("loss_summary", self.loss)

			if not config.is_test:
				with tf.name_scope("Optimization"):
					if config.optimizer_type == 'Adam':
						optimizer = tf.train.AdamOptimizer(config.lr)
					elif config.optimizer_type == 'Gradient':
						optimizer = tf.train.GradientDescentOptimizer(config.lr)
					else :
						optimizer = tf.train.GradientDescentOptimizer(config.lr)
					self.train_op = optimizer.minimize(self.loss, name='train_op')

		with tf.name_scope("Prediction"):
			self.predicted_label = tf.nn.softmax(self.output)
			self.correct_label = self.y_ph
			self.correct_prediction = tf.equal(tf.argmax(self.y_ph,1), tf.argmax(self.output,1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")

			tf.add_to_collection("correct_label", self.y_ph)
			tf.add_to_collection("correct_prediction", self.correct_prediction)
			tf.add_to_collection("accuracy", self.accuracy)
			tf.summary.histogram("minibatch_loss", self.loss)
			tf.summary.histogram("train_prediction", self.predicted_label)

"""#############################################################################################################"""
"""
	ticketClassifier
"""

class TicketClassifierSeqFeatures(object):
	#
	def __init__(self, config, labels):
		self.x_ph = tf.placeholder(shape=[None, None], dtype=tf.int64, name="x")
		self.y_ph = tf.placeholder(shape=[None, None], dtype=tf.int64, name="y")
		self.x_seq_feature_ph = tf.placeholder(shape=[None, 30, 4], dtype=tf.int64, name="seqInputFeature")
		# self.word_embeddings = tf.placeholder(shape=[None, None], dtype=tf.float32, name="word_embeddings")
		self.dropout_keep_prob = tf.placeholder("float", name="dropout_keep_prob")
		self.labels = labels

		with tf.name_scope("Encoder_LSTM"):
			if config.Encoder_type == 'LSTM_NoEmb':
				with tf.name_scope("Encoder_LSTM_1"):
					self.encoder = Encoder(self.x_ph, config)
				with tf.name_scope("Encoder_LSTM_2"):
					self.encoderSeqFeature = EncoderWithNoEmbedding(self.x_seq_feature_ph, config)
				self.concat_encodings = tf.concat((self.encoder.encodings, self.encoderSeqFeature.encodings), axis=1)

		with tf.name_scope("MLP"):
			self.mlp = MLP(self.concat_encodings, config, self.dropout_keep_prob)
			# self.output = self.mlp.one_layer()
			if config.numb_layers == 1:
				self.output, self.W_h_input = self.mlp.linear_layer()
			elif config.numb_layers == 2:
				self.output, self.W_h_input, self.W_h_last = self.mlp.multilayer_perceptron()
			else:
				self.output, self.W_h_input, self.W_h_hidden, self.W_h_last = self.mlp.multilayer_perceptron_2hidden()

		if config.regularization_use:
			with tf.name_scope("Regularization"):
				if config.regularization_type == 'L2':
					if config.numb_layers == 1:
						self.regularizers = tf.nn.l2_loss(self.W_h_input)
					elif config.numb_layers == 2:
						self.regularizers = tf.nn.l2_loss(self.W_h_input) + tf.nn.l2_loss(self.W_h_last)
					else:
						self.regularizers = tf.nn.l2_loss(self.W_h_input) + tf.nn.l2_loss(
							self.W_h_hidden) + tf.nn.l2_loss(self.W_h_last)

				elif config.regularization_use == 'L1':
					if config.numb_layers == 1:
						self.regularizers = config.regularization_beta * tf.reduce_sum(tf.abs(self.W_h_input))
					elif config.numb_layers == 2:
						self.regularizers = config.regularization_beta * (
									tf.reduce_sum(tf.abs(self.W_h_input)) + tf.reduce_sum(tf.abs(self.W_h_last)))
					else:
						self.regularizers = config.regularization_beta * (
									tf.reduce_sum(tf.abs(self.W_h_input)) + tf.reduce_sum(
								tf.abs(self.W_h_hidden)) + tf.reduce_sum(tf.abs(self.W_h_last)))

		with tf.name_scope("Cost"):
			# softmax = tf.nn.softmax(self.output)
			tf.stop_gradient(self.y_ph)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_float(self.y_ph),
																	   logits=self.output)
			self.loss = tf.reduce_mean(cross_entropy)
			# self.loss = tf.losses.log_loss(self.y_ph,self.output, epsilon=0.0001)
			if config.regularization_use:
				self.loss = tf.reduce_mean(self.loss + config.regularization_beta * self.regularizers)

			with tf.device("/cpu:0"):
				self.loss_summary = tf.summary.scalar("loss_summary", self.loss)

			if not config.is_test:
				with tf.name_scope("Optimization"):
					if config.optimizer_type == 'Adam':
						optimizer = tf.train.AdamOptimizer(config.lr)
					elif config.optimizer_type == 'Gradient':
						optimizer = tf.train.GradientDescentOptimizer(config.lr)
					else:
						optimizer = tf.train.GradientDescentOptimizer(config.lr)
					self.train_op = optimizer.minimize(self.loss, name='train_op')

		with tf.name_scope("Prediction"):
			self.predicted_label = tf.nn.softmax(self.output)
			self.correct_label = self.y_ph
			self.correct_prediction = tf.equal(tf.argmax(self.y_ph, 1), tf.argmax(self.output, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")

			tf.add_to_collection("correct_label", self.y_ph)
			tf.add_to_collection("correct_prediction", self.correct_prediction)
			tf.add_to_collection("accuracy", self.accuracy)
			tf.summary.histogram("minibatch_loss", self.loss)
			tf.summary.histogram("train_prediction", self.predicted_label)

	"""#############################################################################################################"""

def runTrainingWithFeatureSequence(config, tickets, targets, labels, ticketsFeaturesSequence):
	#split for cross validation
	tickets_training, targets_training, inputFeature_training, tickets_validation, targets_validation, inputFeature_validation = ut.get_train_testWithSequenceFeatures(tickets, targets, ticketsFeaturesSequence, test_size=0.1)
	print("** Validation Dataset Dimension ** " + str(len(targets_validation)) + " data")
	global best_validation_accuracy

	batcher = bc.Batcher(tickets_training, targets_training, config.batch_size)
	batcher.addSequenceFeatures(inputFeature_training)
	if config.device == "GPU" :
		#config_proto = tf.ConfigProto(device_count=config.device_count)
		config_proto = tf.ConfigProto(allow_soft_placement=True)
	else :
		config_proto = tf.ConfigProto()

	with tf.Session(config=config_proto) as sess:
		tc = TicketClassifierSeqFeatures(config, labels)
		writer = tf.summary.FileWriter(config.tensorboard_path, sess.graph)
		sess.run(tf.global_variables_initializer())

		# Add ops to save and restore all the variables.
		best_saver = tf.train.Saver(max_to_keep=3)
		saver = tf.train.Saver()
		for e in range(config.epochs):
			step = 0
			for batch_x, batch_y, batch_x_sequenceFeature in batcher.batchesWithFeatures():
				_, loss,accuracy,correct_prediction,predicted_label,correct_label = sess.run((tc.train_op,tc.loss,tc.accuracy,tc.correct_prediction,tc.predicted_label, tc.correct_label),feed_dict={tc.x_ph: batch_x,tc.y_ph: batch_y, tc.x_seq_feature_ph: batch_x_sequenceFeature, tc.dropout_keep_prob : 0.8})

				if step%100 == 0 and config.verbose == 'high' :
					print("----------------------------------------")
					print("Epoch :" + str(e))
					print("Step :" + str(step))
					print("Batch Loss         : " + str(loss))
					print("Batch Accuracy     : " + str(accuracy))
					print("Correct Prediction 	: " + str(correct_prediction))

					pl = ut.fromOneHotToTerm(labels, predicted_label)
					cl = ut.fromOneHotToTerm(labels, correct_label)
					print("Predicted Label 	: " + str(pl))
					print("Correct label 	: " + str(cl))
					printTrainingStepInFile(labels,config,loss,accuracy,correct_prediction,predicted_label,correct_label,e,step)

				step += 1
			# Save the variables to disk.
			save_path = saver.save(sess, config.model_path + "model.ckpt", global_step=e)
			print("Model saved in path: %s" % save_path)
			#check on validation set and save if accuracy is better than the best accuracy
			validation_accuracy = sess.run((tc.accuracy),feed_dict={tc.x_ph: tickets_validation, tc.y_ph: targets_validation, tc.x_seq_feature_ph: inputFeature_validation, tc.dropout_keep_prob : 1.0})
			print("Validation Accuracy : " + str(validation_accuracy))
			if e == 0 :
				best_validation_accuracy = validation_accuracy
				best_saver.save(sess, config.best_model_path + "model.ckpt", global_step=e)

			if validation_accuracy > best_validation_accuracy:
				best_saver.save(sess, config.best_model_path + "model.ckpt", global_step=e)
				print("Best saved in path: %s" % save_path)
				best_validation_accuracy = validation_accuracy

			#Epoch Accuracy and loss
			epoch_loss, epoch_accuracy =  sess.run((tc.loss,tc.accuracy),feed_dict={tc.x_ph: tickets_training, tc.y_ph: targets_training, tc.x_seq_feature_ph: inputFeature_training, tc.dropout_keep_prob : 1.0})
			loss_summary = sess.run(tc.loss_summary,feed_dict={tc.x_ph: tickets_training, tc.y_ph: targets_training, tc.x_seq_feature_ph: inputFeature_training, tc.dropout_keep_prob : 1.0})
			print("Epoch Accuracy           : " + str(epoch_accuracy))
			print("Epoch Loss     			: " + str(epoch_loss))
			print("\n")
			writer.add_summary(loss_summary, e)
		writer.close()

	"""#############################################################################################################"""

def runTraining(config, tickets, targets, labels):
	#split for cross validation
	tickets_training, tickets_validation, targets_training, targets_validation = ut.get_train_and_test(tickets, targets, test_size=0.1)
	print("	*** Validation Dataset Dimension ** " + str(len(targets_validation)) + " data")
	global best_validation_accuracy

	batcher = bc.Batcher(tickets_training, targets_training, config.batch_size)

	if config.device == "GPU" :
		#config_proto = tf.ConfigProto(device_count=config.device_count)
		config_proto = tf.ConfigProto(allow_soft_placement=True)
	else :
		config_proto = tf.ConfigProto()

	tf.reset_default_graph()
	with tf.Session(config=config_proto) as sess:
		tc = TicketClassifier(config, labels)
		if config.tensorboard_saving == True:
			writer = tf.summary.FileWriter(config.tensorboard_path, sess.graph)
		sess.run(tf.global_variables_initializer())

		# Add ops to save and restore all the variables.
		best_saver = tf.train.Saver(max_to_keep=config.max_model_checkpoints)
		#if config.transfer_learning load best model and continue traning:
		if config.transfer_learning:
			print("	*** Transfer Learning Enabled\n")
			print("	*** Loading Best Model from : " + config.best_model_path + " \n")
			saver = restoreBestModel(config, sess)
		else:
			saver = tf.train.Saver(max_to_keep=config.max_model_checkpoints)

		for e in range(config.epochs):
			step = 0
			for batch_x, batch_y in batcher.batches():
				_, loss,accuracy,correct_prediction,predicted_label,correct_label = sess.run((tc.train_op,tc.loss,tc.accuracy,tc.correct_prediction,tc.predicted_label, tc.correct_label),feed_dict={tc.x_ph: batch_x,tc.y_ph: batch_y, tc.dropout_keep_prob : 0.8})

				print("----------------------------------------")
				if step%100 == 0:
					print("Step :" + str(step))
					print("Batch Loss         : " + str(loss))
				print("Epoch :" + str(e))
				if config.verbose == 'high':
					print("Batch Accuracy     : " + str(accuracy))
					print("Correct Prediction 	: " + str(correct_prediction))
					pl = ut.fromOneHotToTerm(labels, predicted_label)
					cl = ut.fromOneHotToTerm(labels, correct_label)
					print("Predicted Label 	: " + str(pl))
					print("Correct label 	: " + str(cl))
				#print("\n")
					#printTrainingStepInFile(labels,config,loss,accuracy,correct_prediction,predicted_label,correct_label,e,step)

				step += 1
			# Save the variables to disk.
			save_path = saver.save(sess, config.model_path + "model.ckpt", global_step=e)
			print("Model saved in path: %s" % save_path)
			#check on validation set and save if accuracy is better than the best accuracy
			validation_accuracy = sess.run((tc.accuracy),feed_dict={tc.x_ph: tickets_validation, tc.y_ph: targets_validation, tc.dropout_keep_prob : 1.0})
			print("Validation Accuracy : " + str(validation_accuracy))
			if e == 0 :
				best_validation_accuracy = validation_accuracy
				best_saver.save(sess, config.best_model_path + "model.ckpt", global_step=e)
				print("Best Model saved in path: %s" % config.best_model_path)
			if validation_accuracy > best_validation_accuracy:
				best_saver.save(sess, config.best_model_path + "model.ckpt", global_step=e)
				print("Best Model saved in path: %s" % config.best_model_path)
				best_validation_accuracy = validation_accuracy

			#Epoch Accuracy and loss
			epoch_loss, epoch_accuracy =  sess.run((tc.loss,tc.accuracy),feed_dict={tc.x_ph: tickets_training, tc.y_ph: targets_training, tc.dropout_keep_prob : 1.0})
			loss_summary = sess.run(tc.loss_summary,feed_dict={tc.x_ph: tickets_training, tc.y_ph: targets_training, tc.dropout_keep_prob : 1.0})
			print("Epoch Accuracy           : " + str(epoch_accuracy))
			print("Epoch Loss     			: " + str(epoch_loss))
			print("\n")
			if config.tensorboard_saving :
				writer.add_summary(loss_summary, e)

		if config.tensorboard_saving :
			writer.close()

	"""#############################################################################################################"""

def printTrainingStepInFile(labels,config,loss,accuracy,correct_prediction,predicted_label,correct_label,e,step):
	textfile = None
	my_file = Path(config.training_result_path + 'trainingResults.txt')
	if my_file.is_file():
		textfile = open(config.training_result_path + 'trainingResults.txt', 'a')
	else :
		textfile = open(config.training_result_path + 'trainingResults.txt', 'w+')
	textfile.write("----------------------------- \n")
	textfile.write("Epoch 		 	    : " + str(e) + "\n")
	textfile.write("Step 		 	    : " + str(step) + "\n")
	textfile.write("Batch Loss          : " + str(loss) + "\n")
	textfile.write("Batch Accuracy      : " + str(accuracy) + "\n")
	textfile.write("Correct Prediction 	: " + str(correct_prediction) + "\n")

	pl = ut.fromOneHotToTerm(labels, predicted_label)
	cl = ut.fromOneHotToTerm(labels, correct_label)
	textfile.write("Predicted Label 	: " + str(pl) + "\n")
	textfile.write("Correct label 	    : " + str(cl) + "\n")
	textfile.close()

	return None

"""#############################################################################################################"""
"""Restore Model - Restore del best model"""
def restoreBestModel(config, sess):
	"""restore del Modello"""
	#saver = tf.train.Saver()
	#restore best model
	with sess as sess1:
		saver = tf.train.import_meta_graph(config.best_model_path + config.model_to_restore, clear_devices=True)
		return	saver.restore(sess1, tf.train.latest_checkpoint(config.best_model_path + "/"))

def __restoreBestModel(config, session):
	saver = tf.train.import_meta_graph(config.best_model_path + config.model_to_restore, clear_devices=True)
	saver.restore(session, tf.train.latest_checkpoint(config.best_model_path + "/"))
	graph = tf.get_default_graph()
	return saver, graph

def _restoreBestModel(config, sess):
	with sess as sess1:
		#
		ckpt_state = tf.train.get_checkpoint_state(config.best_model_path, latest_filename='checkpoint_best')
		if platform.system() == 'Windows':
			cp_path = ckpt_state.model_checkpoint_path.replace('\\', '/')
		else:
			cp_path = ckpt_state.model_checkpoint_path
		#
		print("Restoring model...")
		#
		print('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

		saver = tf.train.import_meta_graph(cp_path + '.meta', clear_devices=True)
	return saver.restore(sess1, tf.train.latest_checkpoint(cp_path))

"""#############################################################################################################"""
"""Run Evaluation - Test the model on data"""
def runEvaluation(config, tickets, target, labels, dictionary):
	if config.use_pretrained_embs:
		print("	*** Uso pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		voc = vc.Vocabulary(config)
		reverse_dict = voc.getReverseDictionary(dictionary)
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	with tf.Session() as sess:
		tc = TicketClassifier(config, labels)
		feed_dict={tc.x_ph: tickets, tc.y_ph:target, tc.dropout_keep_prob:1.0}
		print("	*** Loading Best Model from : " + config.best_model_path + " \n")
		saver = restoreBestModel(config, sess)
		#saver = tf.train.Saver()
		#saver.restore(sess, tf.train.latest_checkpoint(config.best_model_path + "/"))

		print("	*** Start Prediction\n")
		output,accuracy = sess.run((tc.predicted_label, tc.accuracy), feed_dict=feed_dict)
		#
		pl = ut.fromOneHotToTerm(labels, output)
		cl = ut.fromOneHotToTerm(labels, target)

		#print(output)
		print("Predicted Class : " + str(pl))
		print("Correct Class : " + str(cl))
		print("Accuracy :" + str(accuracy))
		# print confusion matrix
		metrics = vm.Metrics(target, output, labels)
		confMatrix = metrics.getConfusionMatrix()
		metrics.printConfusionMatrix(confMatrix)

		#
		dataL = dt.Data(config)
		#
		p_k_1 = 0
		p_k_2 = 0
		count_errors = 0
		for i in range(len(tickets)):
			tt = [tickets[i]]
			ty = [target[i]]
			feed_dict = {tc.x_ph: tt, tc.y_ph: ty, tc.dropout_keep_prob: 1.0}
			output = sess.run((tc.predicted_label), feed_dict=feed_dict)
			pl = ut.fromOneHotToTerm(labels, output)
			cl = ut.fromOneHotToTerm(labels, ty)

			if pl != cl and  config.verbose == 'high':
				count_errors = count_errors + 1
				print("------------------------------------------------------")
				print("Predicted: " + str(pl) + " | Real: " + str(cl))
				pll, print_string,mapped,mappedSecond = ut.fromOneHotToTermWithSorting(labels, output)
				print(print_string)
				seqString = dataL.fromSequenceToData(tickets[i], dictionary)
				print("Input Sequence: " + str(seqString))
				print("\n")

				real = cl
				predette = [mapped, mappedSecond]
				val_k1 = metrics.precisionAtK(real, predette, 1)
				val_k2 = metrics.precisionAtK(real, predette, 2)
				p_k_2 = p_k_2 + val_k2
				p_k_1 = p_k_1 + val_k1
			else:
				p_k_2 = p_k_2 + 1
				p_k_1 = p_k_1 + 1

		precision_k1 = p_k_1 / len(tickets)
		precision_k2 = p_k_2 / len(tickets)
		print("Total Errors : " + str(count_errors))
		print("Precision@k1 : " + str(precision_k1))
		print("Precision@k2 : " + str(precision_k2))

"""#############################################################################################################"""
"""Run Prediction - """
def runPrediction(config, tickets, labels, dictionary):
	tf.reset_default_graph()
	y_out = [0] * config.num_classes
	y_out[0] = 1

	with tf.Session() as sess1:
		tc = TicketClassifier(config, labels)
		saver = tf.train.Saver()
		feed_dict = {tc.x_ph: tickets, tc.y_ph: [y_out], tc.dropout_keep_prob: 1.0}
		saver.restore(sess1, tf.train.latest_checkpoint(config.best_model_path + "/"))

		print("	*** Start Prediction\n")
		output, accuracy = sess1.run((tc.predicted_label, tc.accuracy), feed_dict=feed_dict)
		#
		pl = ut.fromOneHotToTerm(labels, output)

	return output, pl

"""#############################################################################################################"""
"""Run Prediction for multiple tickets - """
def runPredictionTickets(config, tickets, labels):
	tf.reset_default_graph()
	targets = []
	target = [1, 0, 0, 0, 0]
	for i in range(len(tickets)):
		targets.append(target)

	tickets_sequence = []
	for ticket in tickets:
		tickets_sequence.append(ticket['sequence'])

	with tf.Session() as sess1:
		tc = TicketClassifier(config, labels)
		saver = tf.train.Saver()
		feed_dict = {tc.x_ph: tickets_sequence, tc.y_ph: targets, tc.dropout_keep_prob: 1.0}
		print("	*** Restore best model " + config.best_model_path)
		saver.restore(sess1, tf.train.latest_checkpoint(config.best_model_path + "/"))
		print("	*** Start Prediction\n")
		output, accuracy = sess1.run((tc.predicted_label, tc.accuracy), feed_dict=feed_dict)
		#
		pl = ut.fromOneHotToTerm(labels, output)

	return output, pl


"""#############################################################################################################"""
def runEvaluationReturnAccuracy(config, tickets, target, labels, dictionary):
	if config.use_pretrained_embs:
		print("	*** Uso pretrained Words Embedding\n")
		skip = sk.SkipgramModel(config)
		skipgramModel = skip.get_skipgram()
		voc = vc.Vocabulary(config)
		reverse_dict = voc.getReverseDictionary(dictionary)
		skipgramEmbedding = skip.getCustomEmbeddingMatrix(skipgramModel, reverse_dict)
		config.skipgramEmbedding = skipgramEmbedding

	tf.reset_default_graph()
	with tf.Session() as sess:
		tc = TicketClassifier(config, labels)
		feed_dict={tc.x_ph: tickets, tc.y_ph:target, tc.dropout_keep_prob:1.0}
		print("	*** Loading Best Model from : " + config.best_model_path + " \n")
		#saver = restoreBestModel(config, sess)
		saver = tf.train.Saver()
		#saver.restore(sess, tf.train.latest_checkpoint(config.best_model_path + "/"))
		#print("	*** Start Prediction\n")
		#output,accuracy = sess.run((tc.predicted_label, tc.accuracy), feed_dict=feed_dict)

		print("	*** Restore best model " + config.best_model_path)
		saver.restore(sess, tf.train.latest_checkpoint(config.best_model_path + "/"))
		print("	*** Start Prediction\n")
		output, accuracy = sess.run((tc.predicted_label, tc.accuracy), feed_dict=feed_dict)
		del tc
		return accuracy
