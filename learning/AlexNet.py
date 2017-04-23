import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import time

import os

###########################################################################################
###                                 TF/NN                                               ###
###########################################################################################
from load import labels
from load.LoadProcess import LoadProcess
from load.MainLoader import MainLoader

class AlexNet():
	def neural_network_model(self, x, is_training: bool = True):
		"""Defines the neural network model. Output is a n_labels long array"""

		#not 100% alexnet, but very alexnet-like

		# input layer
		input = tf.reshape(x, shape=[-1, 224, 224, 1])

		# convolutional layer 1
		conv1 = tf.layers.conv2d(
			inputs=input,
			filters=96,
			kernel_size=11,
			padding="valid",
			strides=4,
			activation=tf.nn.relu,
		)  # [batchsize, 54, 54, 96]

		# max-pooling layer 1
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=3,
			strides=2,
		) #[batchsize, 26, 26, 96]

		# convolutional layer 2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=256,
			kernel_size=5,
			strides=1,
			padding="same",
			activation=tf.nn.relu,
		)  # [batchsize, 26, 26, 256]

		# pooling layer 2
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=3,
			strides=2,
		)  # [batchsize, 12, 12, 256]

		conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=384,
			kernel_size=3,
			strides=1,
			padding="same",
			activation=tf.nn.relu,
		) #[12,12, 384]

		conv4 = tf.layers.conv2d(
			inputs=conv3,
			filters=384,
			kernel_size=3,
			strides=1,
			padding="same",
		) #[ , 12, 12, 384 ]

		conv5 = tf.layers.conv2d(
			inputs=conv4,
			filters=256,
			kernel_size=3,
			strides=1,
			padding="same",
			activation=tf.nn.relu,
		) #[, 12, 12, 256]

		pool3 = tf.layers.max_pooling2d(
			inputs=conv5,
			pool_size=3,
			strides=2,
		)#[, 5, 5, 256]

		pool3flattened = tf.reshape(pool3, [-1, 5*5*256])

		#fully connected
		fc1 = tf.layers.dense(
			inputs=pool3flattened,
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
			units=1000,
			activation=tf.nn.relu,
		)

		# add dropout
		dropout = tf.layers.dropout(
			inputs=fc3,
			rate=self.dropout_rate,
			training=is_training,
		)

		output = tf.layers.dense(
			inputs=dropout,
			units=self.n_classes,
		)

		return output


	# def next_batch_pipe(loader: MainLoader, batch_size: int, images_used: int, is_training: bool, conn):
	# 	x, y = loader.next_batch(batch_size, images_used, is_training)
	# 	conn.send((x, y))
	# 	conn.close()


	def train_neural_network(self, x):
		accs = []
		epochs = []
		print("start neural network training")

		nn_output = self.neural_network_model(x)

		# cost function
		cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=self.y))

		# optimizer function
		optimizer_func = tf.train.GradientDescentOptimizer(self.lr).minimize(cost_func)
		# init variables and session
		init = tf.global_variables_initializer()



		process = LoadProcess(self.loader, self.batch_size, self.image_load_size, self.size, self.n_classes)
		process.runtraining()


		lognum = 10

		with tf.Session() as sess:
			sess.run(init)
			saver = tf.train.Saver()
			t_total = time.time()
			for epoch in range(self.num_epochs):
				epoch_cost = 0
				t_batch_start = time.time()
				for batch_num in range(self.num_train_batches):


					#todo: join

					# t_load_start = time.time()
					process.wait()
					# t_load = time.time() - t_load_start
					# x_b = np.frombuffer(x_arr_batch.get_obj()).reshape(batch_shape)
					# y_b = np.frombuffer(y_arr_batch.get_obj()).reshape(labels_shape)
					# t_batch_start = time.time()
					x_b, y_b = process.get_batch()
					# t_batch = time.time() - t_batch_start
					#todo: copy ... or not

					# t_train_start = time.time()
					batch, c = sess.run([optimizer_func, cost_func], feed_dict={x: x_b, self.y: y_b})
					# t_train  = time.time() - t_train_start

					#todo: start
					# t_run_start = time.time()
					process.runtraining()
					# t_run = time.time() - t_run_start
					# t_tot = time.time()

					epoch_cost += c
					if(batch_num % lognum == 0):
						print(lognum, 'batches took', '{:10.3f}'.format(time.time() - t_total), 'seconds')
						t_total = time.time()

						# print("Batch ", batch_num, " of ", self.num_train_batches, "\tCost ", "{:10.6f}".format(c),
						#       " previous batch wait time", "{:10.2f}".format(t_train),
						#       ' previous batch loading time', "{:10.2f}".format(t_load),
						#       ' get time', "{:10.2f}".format(t_batch),
						#       ' run time', "{:10.2f}".format(t_run),
						#       ' batch time ', "{:10.2f}".format(t_tot - t_load_start))
					# if (batch_num % 500 == 0 and batch_num != 0):
					# 	saver.save(sess, base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "batch" + str(batch_num) + "cost" + "{:0.2f}".format(c) + ".checkpoint")

				print("Epoch", str(epoch), " of ", str(self.num_epochs), " cost: ", str(epoch_cost), 'Time: ', time.time() - t_batch_start)
				correct = tf.equal(tf.argmax(nn_output, 1), tf.argmax(self.y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, "float"))

				print("Epoch complete. Calculating accuracy...")

				epoch_acc = 0
				for n in range(self.num_test_batches):
					t0 = time.time()
					test_batch_x, test_batch_y = self.loader.next_batch(self.batch_size, is_training=False)
					#test_batch_x, test_batch_y = mnist.test.next_batch(batch_size)
					epoch_acc += accuracy.eval({x: test_batch_x, self.y: test_batch_y})
					print("Calculating accuracy. ", "{:10.2f}".format((n / self.num_test_batches) * 100), "% complete. Time:", (time.time() - t0))
				acc = epoch_acc / self.num_train_batches
				print("Epoch Accuracy: ", acc, 'Epoch time:', time.time() - t_batch_start, 'Total time:', time.time() - t_total)
				accs.append(acc)
				epochs.append(epoch)

				if epoch % 5 == 0:
					saver.save(sess, self.base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(acc) + ".checkpoint")
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



	def run_nn(self, x, epoch, acc):
		"""Runs a pre-trained network. x is a flattened image of the same size as the model has been trained"""
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph(self.base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(acc) + ".checkpoint.meta")
			saver.restore(sess, self.base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(
				acc) + ".checkpoint")

	def __init__(self):

		os.environ["CUDA_VISIBLE_DEVICES"] = "0"


		# print(device_lib.list_local_devices())

		###########################################################################################
		###                                 THINGS                                              ###
		###########################################################################################

		self.base_dir = os.path.dirname(os.path.dirname(__file__))

		# data loader
		self.test_set_rate = 0.01  # fraction of dataset used as test-set
		self.loader = MainLoader(224, self.test_set_rate)
		print("loader initialized")

		# data things
		self.batch_size = 100
		self.image_load_size = 2
		self.test_size = len(self.loader.data) * (self.test_set_rate)
		self.num_train_batches = int(np.ceil(len(self.loader.trainindexes) / self.image_load_size))
		self.num_test_batches = int(np.ceil(len(self.loader.testindexes) / self.batch_size))
		# training things
		self.num_epochs = 10
		self.dropout_rate = 0.2
		self.lr = 0.001

		# classifier things
		self.size = 224  # (X * X size)
		self.n_classes = len(labels)
		self.flat_batch_size = self.size * self.size * self.batch_size
		self.flat_labels_size = self.batch_size * self.n_classes
		self.batch_shape = (self.batch_size, self.size * self.size)
		self.labels_shape = (self.batch_size, self.n_classes)

		# tensorflow things
		self.x = tf.placeholder("float", [None, 224 * 224])
		self.y = tf.placeholder("float")
		self.train_neural_network(self.x)

# if __name__ == '__main__':

# run_nn(x, 0, 0.01)
