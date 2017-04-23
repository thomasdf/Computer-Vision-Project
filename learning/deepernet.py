import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import time

import os

from load import labels
from load.LoadProcess import LoadProcess
from load.MainLoader import MainLoader


class DeeperNet():
	####################################################################
	###                                 Helpers                      ###
	####################################################################
	def conv2D(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

	def maxpool2d(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	def print_batch_info(self, batch_num, num_batches, time):
		print("Batch ", batch_num, " of ", num_batches, " complete. Time: ", "{: 1.2f}".format(time))

	def print_epoch_info(self, epoch_num, num_epochs, loss):
		print("Epoch ", epoch_num, " of ", num_epochs, " complete. loss: ", loss)

	def print_acc_info(self, accuracy):
		print("accuracy: ", accuracy)

	def print_acc_progress(self, percentage):
		print("calculating accuracy. ", percentage, "% complete.")

	def accuracy(self, prediction):
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
		acc = tf.reduce_mean(tf.cast(correct, "float"))
		accumulated_acc = 0
		for i in range(self.num_test_batches):
			test_x, test_y = self.loader.next_batch(self.batch_size, is_training=False)
			accumulated_acc += acc.eval({self.x: test_x, self.y: test_y})
			if i % max((self.num_test_batches // 10), 1) == 0:
				self.print_acc_progress(i * 10)
		self.print_acc_info(accumulated_acc / self.num_test_batches)
		self.loader.index_test = 0
		return accumulated_acc

	####################################################################
	###                               NN-Definition                  ###
	####################################################################

	def neural_network_model(self, x, is_training: bool = True):
		"""Defines the neural network model. Output is a n_labels long array"""

		# input layer
		input = tf.reshape(x, shape=[-1, int(self.size), int(self.size), 1])

		# convolutional layer 1
		conv1 = tf.layers.conv2d(
			inputs=input,
			filters=64,
			kernel_size=5,
			padding="same",
			activation=tf.nn.relu,
		)

		# max-pooling layer 1
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=2,
			strides=2,
		)

		# convolutional layer 2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=128,  # output: [batchsize, input, input, 64]
			kernel_size=5,
			padding="same",  # output of same size as input
			activation=tf.nn.relu,
		)  # [batchsize, 14, 14, 64]

		# pooling layer 2
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=2,
			strides=2,
		)  # [batchsize, 7, 7, 64]

		# flatten:
		pool2flattened = tf.reshape(pool2, [-1, int(self.size / 4 * self.size / 4 * 128)])

		# fully connected layers:
		fc1 = tf.layers.dense(
			inputs=pool2flattened,
			units=4096,
			activation=tf.nn.relu,
		)

		fc2 = tf.layers.dense(
			inputs=fc1,
			units=4096,
			activation=tf.nn.relu,
		)

		fc3 = tf.layers.dense(
			inputs=fc2,
			units=1024,
			activation=tf.nn.relu,
		)

		# add dropout
		dropout = tf.layers.dropout(
			inputs=fc3,
			rate=self.dropout_rate,
			training=is_training,
		)  # [batchsize, 1024]

		logits = tf.layers.dense(
			inputs=dropout,
			units=self.n_classes,
		)

		return logits

	####################################################################
	###                               Training                       ###
	####################################################################

	def train_neural_network(self):
		x = self.x
		accs = []
		epochs = []
		print("start neural network training")

		logits = self.neural_network_model(x)

		# cost function
		cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

		# optimizer function
		optimizer_func = tf.train.AdamOptimizer(self.lr).minimize(cost_func)
		# init variables and session
		init = tf.global_variables_initializer()

		process = LoadProcess(self.loader, self.batch_size, self.image_load_size, self.size, self.n_classes)
		process.runtraining()

		batch_lognum = 10  # batch printing/batch modulo
		batch_accuracy_modulo = 50
		epoch_lognum = 1  # epoch info interval
		epoch_save_modulo = 1  # save interval
		plot_modulo = 5  # plot interval

		with tf.Session() as sess:
			sess.run(init)
			saver = tf.train.Saver()
			t_total = time.time()

			self.accuracy(logits)  # test accuracy of random network

			for epoch in range(self.num_epochs):
				epoch_loss = 0
				t_batch_start = time.time()
				for batch_num in range(self.num_train_batches):

					# join
					process.wait()

					x_b, y_b = process.get_batch()

					_, c = sess.run([optimizer_func, cost_func], feed_dict={x: x_b, self.y: y_b})

					# start
					process.runtraining()

					epoch_loss += c
					if (batch_num % batch_lognum == 0):
						self.print_batch_info(batch_num, self.num_train_batches, time.time() - t_total)
						t_total = time.time()

					if batch_num % batch_accuracy_modulo == 0 and batch_num != 0:
						self.accuracy(logits)

				if epoch % epoch_lognum == 0:
					self.print_epoch_info(epoch, self.num_epochs, epoch_loss)

				print("Epoch complete. Calculating accuracy...")

				acc = self.accuracy(logits)
				accs.append(acc)
				epochs.append(epoch)

				if epoch % epoch_save_modulo == 0:
					saver.save(sess,
					           self.base_dir + "/savedmodels/thomasnet/epoch" +
					           str(epoch) + "acc" + "{:1.3f}".format(acc) + ".checkpoint")

				if epoch % plot_modulo == 0 and epoch != 0:
					plt.figure()
					gen, = plt.plot(epochs, accs, label='accuracy vs epoch')
					plt.legend()
					plt.show()

				self.loader.reset_index()
			print('Total time spent', time.time() - t_total)
			plt.figure()
			gen = plt.plot(epochs, accs, label='accuracy vs epoch')
			plt.legend()
			plt.show()

		####################################################################
		###                               Running                        ###
		####################################################################

	def run_nn(self, batch, epoch, acc):
		"""Runs a pre-trained network. x is a flattened image of the same size as the model has been trained"""

		logits = self.neural_network_model(self.x, False)
		chachpoint_path = self.base_dir + "/savedmodels/thomasnet/epoch" + str(epoch) + "acc" + "{:1.3f}".format(acc) + '.checkpoint'
		saver = tf.train.Saver()
		with tf.Session() as sess:
			# saver = tf.train.import_meta_graph(chachpoint_path + '.meta')
			saver.restore(sess=sess, save_path=chachpoint_path)
			all_vars = tf.get_collection('vars')
			for v in all_vars:
				sess.run(v)
			# init = tf.global_variables_initializer()
			# sess.run(init)
			res = sess.run(logits, feed_dict={self.x: batch})
		return res

	####################################################################
	###                               class-things                   ###
	####################################################################

	def __init__(self):
		# print(device_lib.list_local_devices())
		self.base_dir = os.path.dirname(os.path.dirname(__file__))

		# Variables
		# classifier
		self.size = 32  # (X * X size)
		self.num_epochs = 10
		self.dropout_rate = 0.2
		self.lr = 1e-7

		# loader
		self.batch_size = 500
		self.image_load_size = 25
		self.test_set_rate = 0.05  # fraction of dataset used as test-set
		self.dataset_fraction = 0.02  # fraction of whole dataset used

		# data loader
		self.loader = MainLoader(self.size, self.test_set_rate)
		print("loader initialized")

		# classifier things
		self.n_classes = len(labels)
		self.flat_batch_size = self.size * self.size * self.batch_size
		self.flat_labels_size = self.batch_size * self.n_classes
		self.batch_shape = (self.batch_size, self.size * self.size)
		self.labels_shape = (self.batch_size, self.n_classes)

		# data things

		self.test_size = len(self.loader.data) * (self.test_set_rate)
		self.num_train_batches = int(
			(np.ceil(len(self.loader.trainindexes) / self.image_load_size)) * self.dataset_fraction)
		self.num_test_batches = max(
			int((np.ceil(len(self.loader.testindexes) / self.batch_size)) * self.dataset_fraction), 1)
		# training things

		# tensorflow things
		self.x = tf.placeholder("float", [None, self.size * self.size])
		self.y = tf.placeholder("float")
