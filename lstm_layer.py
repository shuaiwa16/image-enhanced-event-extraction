import tensorflow as tf
import os
import numpy as np
#import data_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MAX_ITER = 1000
MODEL_DIR = 'model/en.ckpt'

class Emd_layer:
	def __init__(self, dim = 100, name = 'emb_layer'):
		self.dim = dim
		self.name = name

	def __call__(self, x_input, embeddings, reuse = False, keep_prob = 0.5):
		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
				keep_prob = 1.0
			embedding_var = tf.get_variable(name = 'emb', shape = embeddings.shape, 
				dtype = tf.float32, initializer = tf.constant_initializer(embeddings))
			input_vec = tf.nn.embedding_lookup(embedding_var, x_input)
			input_vec = tf.nn.dropout(input_vec, keep_prob = keep_prob)
		return input_vec

class Lstm_layer:
	def __init__(self, dim = 100, name = 'lstm_layer'):
		self.dim = dim
		self.name = name

	def __call__(self, x_input, x_length, reuse = False, keep_prob = 0.5):
		with tf.variable_scope(self.name) as vs:
			if reuse == True:
				vs.reuse_variables()
				keep_prob = 1.0
			f_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.dim)
			f_cell = tf.nn.rnn_cell.DropoutWrapper(f_cell, output_keep_prob = keep_prob)
			#mf_cell = tf.nn.rnn_cell.MultiRNNCell([f_cell] * 2)
			b_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.dim)
			b_cell = tf.nn.rnn_cell.DropoutWrapper(b_cell, output_keep_prob = keep_prob)
			#mb_cell = tf.nn.rnn_cell.MultiRNNCell([b_cell] * 2)
			outputs, state = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, 
				x_input, x_length, dtype = tf.float32)
			f_state = state[0]
			b_state = state[1]
			doc_vec = tf.concat([f_state[0], b_state[0]], 
				axis = 1)
			f_outputs = outputs[0]
			b_outputs = outputs[1]
			f_outputs = tf.reshape(f_outputs, shape = [-1, 10000])
			b_outputs = tf.reshape(b_outputs, shape = [-1, 10000])
			outputs = tf.concat([f_outputs, b_outputs], axis = 1)
		return doc_vec

class Fc_layer:
	def __init__(self, lay_num = 2, dim = [20, 2], name = 'fc_layer'):
		self.lay_num = lay_num
		self.dim = dim
		self.name = name

	def __call__(self, x_input, reuse = False, keep_prob = 0.5):
		with tf.variable_scope(self.name) as vs:
			if reuse == True:
				vs.reuse_variables()
				keep_prob = 1.0
			outputs = x_input
			for i, d in enumerate(self.dim):
				if i == self.lay_num - 1:
					outputs = tf.layers.dense(outputs, units = d)
				else:
					outputs = tf.layers.dense(outputs, units = d, 
						activation = tf.nn.relu)
					outputs = tf.nn.dropout(outputs, keep_prob = keep_prob)
		return outputs

def split_data(data_batch):
	num = len(data_batch)
	inputs = np.asarray([item[:100] for item in data_batch])
	lengths = np.asarray([item[100] for item in data_batch])
	labels = np.zeros([num, 2])
	for i in range(num):
		labels[i][data_batch[i][101]] = 1.0

	return inputs, lengths, labels

def split_soft_data(data_batch):
	num = len(data_batch)
	inputs = np.asarray([item[:100] for item in data_batch])
	inputs = np.cast[np.int32](inputs)
	lengths = np.asarray([item[100] for item in data_batch])
	lengths = np.cast[np.int32](lengths)
	labels = np.asarray([item[101:] for item in data_batch])

	return inputs, lengths, labels

def train_english():
	current_best = 0.0
	current_index = -1
	for i in range(MAX_ITER):
		english_data_batch = data_utils.sample_batch()
		english_real_inputs, english_real_lengths, english_real_labels = split_soft_data(english_data_batch)
		_, english_loss_val = sess.run([english_opt, english_loss], 
			feed_dict = {english_input:english_real_inputs, 
			english_length:english_real_lengths, 
			english_labels:english_real_labels})
		if i % 20 == 0:
			training_accuracy = sess.run(english_accuracy, 
				feed_dict= {english_input:english_real_inputs, 
				english_length:english_real_lengths, 
				english_labels:english_real_labels})
			test_accuracy = 0.0
			for j in range(30):
				english_data_batch = data_utils.sample_batch(is_training = False, test_id = j)
				english_real_inputs, english_real_lengths, english_real_labels = split_data(english_data_batch)
				test_accuracy = test_accuracy + sess.run(english_accuracy_test, 
					feed_dict = {english_input:english_real_inputs, 
					english_length:english_real_lengths, 
					english_labels:english_real_labels})
			test_accuracy = test_accuracy / 30.0
			if test_accuracy > current_best:
				current_index = i
				current_best = test_accuracy
				print ('saving...')
				saver.save(sess, MODEL_DIR, global_step = i)
			print ('iter %d, training loss %f, training accuracy %f, test accuracy %f' %
				(i, english_loss_val, training_accuracy, test_accuracy))

def inference():
	ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
	if ckpt and ckpt.model_checkpoint_path:
		print (ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	end_flag = False
	i = 0
	soft_labels = []
	while end_flag == False:
		parl_batch, end_flag = data_utils.parl_sample_batch(i = i)
		parl_inputs, parl_lengths, _ = split_data(parl_batch)
		english_soft_label = sess.run(english_output_test_softmax, 
			feed_dict = {english_input:parl_inputs, english_length:parl_lengths})
		for vec in english_soft_label:
			soft_labels.append(vec)
		i = i + 1
	return soft_labels

if __name__ == '__main__':
	vocab_dir = os.path.join(data_utils.FR_DATA_DIR, 'vocab.txt')
	mat_dir = os.path.join(data_utils.FR_DATA_DIR, 'emb.npy')
	train_dir = os.path.join(data_utils.FR_DATA_DIR, 'book/train/dataset.txt')
	test_dir = os.path.join(data_utils.FR_DATA_DIR, 'book/test/dataset.txt')
	parl_doc_dir = os.path.join(data_utils.FR_DATA_DIR, 'book')
	word2id, embeddings = data_utils.english_initialize(vocab_dir, mat_dir, train_dir, test_dir)
	#data_utils.parallel_initialize(parl_doc_dir)
	data_utils.soft_label_initialize(parl_doc_dir)

	english_input = tf.placeholder(dtype = tf.int32, shape = [None, data_utils.MAX_LEN])
	english_labels = tf.placeholder(dtype = tf.float32, shape = [None, 2])
	english_length = tf.placeholder(dtype = tf.int32, shape = [None])

	english_embedding_layer = Emd_layer(name = 'english_embedding_layer')
	english_lstm_layer = Lstm_layer(name = 'english_lstm_layer')
	english_fc_layer = Fc_layer(name = 'english_fc_layer')

	english_input_vec = english_embedding_layer(english_input, embeddings)
	english_input_vec_test = english_embedding_layer(english_input, embeddings, 
		reuse = True, keep_prob = 1.0)
	english_repre_vec = english_lstm_layer(english_input_vec, english_length)
	english_repre_vec_test = english_lstm_layer(english_input_vec_test, english_length, 
		reuse = True, keep_prob = 1.0)
	english_output = english_fc_layer(english_repre_vec)
	english_output_test = english_fc_layer(english_repre_vec_test, reuse = True)
	english_output_test_softmax = tf.nn.softmax(english_output_test)

	english_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = english_labels, 
		logits = english_output))
	english_opt = tf.train.AdamOptimizer().minimize(english_loss)
	english_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(english_output, 
		axis = 1), tf.argmax(english_labels, axis = 1)), tf.float32))
	english_accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(english_output_test, 
		axis = 1), tf.argmax(english_labels, axis = 1)), tf.float32))

	'''
	french_input = tf.placeholder(dtype = tf.int32, shape = [None, data_utils.MAX_LEN])
	french_labels = tf.placeholder(dtype = tf.float32, shape = [None, 2])
	french_length = tf.placeholder(dtype = tf.int32, shape = [None])

	french_embedding_layer = Emd_layer(name = 'french_embedding_layer')
	french_lstm_layer = Lstm_layer(name = 'french_lstm_layer')
	french_fc_layer = Fc_layer(name = 'french_fc_layer')

	french_input_vec = french_embedding_layer(french_input, embeddings)
	french_input_vec_test = french_embedding_layer(french_input, embeddings, 
		reuse = True, keep_prob = 1.0)
	french_repre_vec = french_lstm_layer(french_input_vec, french_length)
	french_repre_vec_test = french_lstm_layer(french_input_vec_test, french_length, 
		reuse = True, keep_prob = 1.0)
	french_output = french_fc_layer(french_repre_vec)
	french_output_test = french_fc_layer(french_repre_vec_test, reuse = True)
	french_output_test_softmax = tf.nn.softmax(french_output_test)
	'''
	
	saver = tf.train.Saver()
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		train_english()
		#soft_labels = inference()
		#data_utils.soft_label_assign(soft_labels, parl_doc_dir)










