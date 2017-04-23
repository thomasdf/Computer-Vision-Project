import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import time

import os


from load import labels
from load.LoadProcess import LoadProcess
from load.MainLoader import MainLoader

class ThomasNet():
####################################################################
###                                 Helpers                      ###
####################################################################
	def conv2D(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

	def maxpool2d(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


	def print_batch_info(self, batch_num, num_batches):
		print("Batch ", batch_num, " of ", num_batches, " complete.")


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
			if i % max((self.num_test_batches // 10),1) == 0:
				self.print_acc_progress(i * 10)
		self.print_acc_info(accumulated_acc / 5)
		self.loader.index_test = 0
		return accumulated_acc

####################################################################
###                               NN-Definition                  ###
####################################################################

	def neural_network_model(self, x, is_training: bool = True):
		"""Defines the neural network model. Output is a n_labels long array"""

		# define variables for layers (that is: allocate memory, create a structure)
		weights = {"conv1": tf.Variable(tf.random_normal([5, 5, 1, 64])),
		           "conv2": tf.Variable(tf.random_normal([5, 5, 64, 128])),
		           "fc": tf.Variable(tf.random_normal([int(self.size / 4 * self.size / 4 * 128), 4096])),
		           "out": tf.Variable(tf.random_normal([4096, self.n_classes]))
		           }

		biases = {"conv1": tf.Variable(tf.random_normal([64])),
		          "conv2": tf.Variable(tf.random_normal([128])),
		          "fc": tf.Variable(tf.random_normal([4096])),
		          "out": tf.Variable(tf.random_normal([self.n_classes]))
		          }

		x = tf.reshape(x, shape=[-1, self.size, self.size, 1])

		conv1 = tf.nn.relu(self.conv2D(x, weights["conv1"]) + biases["conv1"])
		conv1 = self.maxpool2d(conv1)

		conv2 = tf.nn.relu(self.conv2D(conv1, weights["conv2"]) + biases["conv2"])
		conv2 = self.maxpool2d(conv2)

		fc = tf.reshape(conv2, [-1, int(self.size / 4 * self.size / 4 * 128)])
		fc = tf.nn.relu(tf.matmul(fc, weights["fc"]) + biases["fc"])

		# fc = tf.nn.dropout(fc, keep_rate)

		output = tf.matmul(fc, weights["out"]) + biases["out"]

		return output

	# def next_batch_pipe(loader: MainLoader, batch_size: int, images_used: int, is_training: bool, conn):
	# 	x, y = loader.next_batch(batch_size, images_used, is_training)
	# 	conn.send((x, y))
	# 	conn.close()

####################################################################
###                               Training                       ###
####################################################################

	def train_neural_network(self, x):
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
		epoch_lognum = 1  # epoch info interval
		epoch_save_modulo = 1 #save interval
		plot_modulo = 5 #plot interval

		with tf.Session() as sess:
			sess.run(init)
			saver = tf.train.Saver()
			t_total = time.time()

			self.accuracy(logits) #test accuracy of random network

			for epoch in range(self.num_epochs):
				epoch_loss = 0
				t_batch_start = time.time()
				for batch_num in range(self.num_train_batches):

					# todo: join

					# t_load_start = time.time()
					process.wait()
					# t_load = time.time() - t_load_start
					# x_b = np.frombuffer(x_arr_batch.get_obj()).reshape(batch_shape)
					# y_b = np.frombuffer(y_arr_batch.get_obj()).reshape(labels_shape)
					# t_batch_start = time.time()
					x_b, y_b = process.get_batch()
					# t_batch = time.time() - t_batch_start
					# todo: copy ... or not

					# t_train_start = time.time()
					_, c = sess.run([optimizer_func, cost_func], feed_dict={x: x_b, self.y: y_b})
					# t_train  = time.time() - t_train_start

					# todo: start
					# t_run_start = time.time()
					process.runtraining()
					# t_run = time.time() - t_run_start
					# t_tot = time.time()

					epoch_loss += c
					if (batch_num % batch_lognum == 0):
						self.print_batch_info(batch_num, self.num_train_batches)
						print(batch_lognum, 'batches took', '{:10.3f}'.format(time.time() - t_total), 'seconds')
						t_total = time.time()

					# print("Batch ", batch_num, " of ", self.num_train_batches, "\tCost ", "{:10.6f}".format(c),
					#       " previous batch wait time", "{:10.2f}".format(t_train),
					#       ' previous batch loading time', "{:10.2f}".format(t_load),
					#       ' get time', "{:10.2f}".format(t_batch),
					#       ' run time', "{:10.2f}".format(t_run),
					#       ' batch time ', "{:10.2f}".format(t_tot - t_load_start))
					# if (batch_num % 500 == 0 and batch_num != 0):
					#

				if epoch % epoch_lognum == 0:
					self.print_epoch_info(epoch, self.num_epochs, epoch_loss)

				print("Epoch complete. Calculating accuracy...")

				acc = self.accuracy(logits)
				accs.append(acc)
				epochs.append(epoch)

				if epoch % epoch_save_modulo == 0:
					saver.save(sess, self.base_dir + "/savedmodels/thomasnet/epoch" + str(epoch) + "acc" + "{:1.3f}".format(
						acc) + ".checkpoint")

				if epoch % plot_modulo == 0:
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

		with tf.Session() as sess:
			saver = tf.train.import_meta_graph(
				self.base_dir + "/savedmodels/thomasnet/epoch" + str(epoch) + "acc" + "{:1.3f}".format(
					acc) + ".checkpoint.meta")
			saver.restore(sess, self.base_dir + "/savedmodels/thomasnet/epoch" + str(epoch) + "acc" + "{:1.3f}".format(
				acc) + ".checkpoint")

			init = tf.global_variables_initializer()
			sess.run(init)
			res = sess.run(tf.nn.softmax(logits), feed_dict={self.x: batch})
		return res

####################################################################
###                               class-things                   ###
####################################################################

	def __init__(self):

		os.environ["CUDA_VISIBLE_DEVICES"] = "0"

		# print(device_lib.list_local_devices())
		self.base_dir = os.path.dirname(os.path.dirname(__file__))

		# Variables
		#classifier
		self.size = 72  # (X * X size)
		self.num_epochs = 10
		self.dropout_rate = 0.2
		self.lr = 1e-5

		#loader
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
		self.num_train_batches = int((np.ceil(len(self.loader.trainindexes) / self.image_load_size))* self.dataset_fraction)
		self.num_test_batches = max(int((np.ceil(len(self.loader.testindexes) / self.batch_size)) * self.dataset_fraction), 1)
		# training things


		# tensorflow things
		self.x = tf.placeholder("float", [None, self.size * self.size])
		self.y = tf.placeholder("float")
		self.train_neural_network(self.x)
