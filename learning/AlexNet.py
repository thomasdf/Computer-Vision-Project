import multiprocessing

import functools
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import ctypes

import time

# from multiprocessing import freeze_support
# from tensorflow.python.client import device_lib
# from load.MainLoader import labels

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from PIL import Image

from async.LoadBatch import next_batch_queue, next_batch_queue_2, next_batch_queue_3, next_batch_arr
from load import labels
from load.MainLoader import MainLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(device_lib.list_local_devices())

###########################################################################################
###                                 THINGS                                              ###
###########################################################################################

base_dir = os.path.dirname(os.path.dirname(__file__))

# data loader
test_set_rate = 0.01  # fraction of dataset used as test-set
loader = MainLoader(224, test_set_rate)
print("loader initialized")

# data things
batch_size = 10
image_load_size = batch_size // 2
test_size = len(loader.data) * (test_set_rate)
num_train_batches = int(np.ceil(len(loader.trainindexes) / image_load_size))
num_test_batches =  int(np.ceil(len(loader.testindexes) / batch_size))
# training things
num_epochs = 10
dropout_rate = 0.2
lr = 0.001

# classifier things
size = 224 # (X * X size)
n_classes = len(labels)
flat_batch_size = size* size * batch_size
flat_labels_size = batch_size*n_classes
batch_shape = (batch_size, size*size)
labels_shape = (batch_size, n_classes)


# tensorflow things
x = tf.placeholder("float", [None, 224*224])
y = tf.placeholder("float")

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#n_classes = 10


###########################################################################################
###                                 TF/NN                                               ###
###########################################################################################

def neural_network_model(x, is_training: bool = True):
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
		rate=dropout_rate,
		training=is_training,
	)

	output = tf.layers.dense(
		inputs=dropout,
		units=n_classes,
	)

	return output


# def next_batch_pipe(loader: MainLoader, batch_size: int, images_used: int, is_training: bool, conn):
# 	x, y = loader.next_batch(batch_size, images_used, is_training)
# 	conn.send((x, y))
# 	conn.close()


def train_neural_network(x):
	accs = []
	epochs = []
	print("start neural network training")

	nn_output = neural_network_model(x)

	# cost function
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=y))

	# optimizer function
	optimizer_func = tf.train.GradientDescentOptimizer(lr).minimize(cost_func)
	# init variables and session
	init = tf.global_variables_initializer()



	x_arr_batch = multiprocessing.Array(ctypes.c_double, flat_batch_size)
	y_arr_batch = multiprocessing.Array(ctypes.c_double, flat_labels_size)

	train_process = multiprocessing.Process(target=loader.next_batch_async_arr,
	                                        args=(batch_size, image_load_size, True, x_arr_batch, y_arr_batch))

	train_process.start()
	# train_process.join()
	#
	# x_b = np.frombuffer(x_arr_batch.get_obj()).reshape(batch_shape)
	# y_b = np.frombuffer(y_arr_batch.get_obj()).reshape(labels_shape)

	# xx_b = x_b.reshape(batch_shape)
	# yy_b = y_b.reshape(labels_shape)

	# x_0 = xx_b[0].reshape((size, size))
	# x_0.astype(np.float32)
	# arr = np.multiply(x_0, 255.0)
	#
	# Image.fromarray(arr).show()


	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver()
		t_total = time.time()
		for epoch in range(num_epochs):
			epoch_cost = 0
			t_batch_start = time.time()
			for batch_num in range(num_train_batches):
				t_load_start = time.time()
				t_load_end = time.time()
				t_train_start = time.time()

				#todo: join

				train_process.join()

				x_b = np.frombuffer(x_arr_batch.get_obj()).reshape(batch_shape)
				y_b = np.frombuffer(y_arr_batch.get_obj()).reshape(labels_shape)

				#todo: copy ... or not

				batch, c = sess.run([optimizer_func, cost_func], feed_dict={x: x_b, y: y_b})

				#todo: start

				train_process = multiprocessing.Process(target=loader.next_batch_async_arr,
				                                        args=(batch_size, image_load_size, True, x_arr_batch, y_arr_batch))

				train_process.start()

				t_train_end  = time.time()
				epoch_cost += c
				if(batch_num % 100 == 0):
					print("Batch ", batch_num, " of ", num_train_batches, "\tCost ", "{:10.6f}".format(c), "\tprevious batch training time",
					      "{:10.2f}".format(t_train_end - t_train_start), '\tprevious batch loading time',
					      "{:10.2f}".format(t_load_end - t_load_start))
				#save each x batches
				# if (batch_num % 500 == 0 and batch_num != 0):
				# 	saver.save(sess, base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "batch" + str(batch_num) + "cost" + "{:0.2f}".format(c) + ".checkpoint")

			print("Epoch", str(epoch), " of ", str(num_epochs), " cost: ", str(epoch_cost), 'Time: ', time.time() - t_batch_start)
			correct = tf.equal(tf.argmax(nn_output, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))

			print("Epoch complete. Calculating accuracy...")

			epoch_acc = 0
			for n in range(num_test_batches):
				t0 = time.time()
				test_batch_x, test_batch_y = loader.next_batch(batch_size, is_training=False)
				#test_batch_x, test_batch_y = mnist.test.next_batch(batch_size)
				epoch_acc += accuracy.eval({x: test_batch_x, y: test_batch_y})
				print("Calculating accuracy. ", "{:10.2f}".format((n / num_test_batches) * 100), "% complete. Time:", (time.time() - t0))
			acc = epoch_acc / num_train_batches
			print("Epoch Accuracy: ", acc, 'Epoch time:', time.time() - t_batch_start, 'Total time:', time.time() - t_total)
			accs.append(acc)
			epochs.append(epoch)

			if epoch % 5 == 0:
				saver.save(sess, base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(acc) + ".checkpoint")
				plt.figure()
				gen, = plt.plot(epochs, accs, label='accuracy vs epoch')
				plt.legend()
				plt.show()
			loader.reset_index()
		print('Total time spent', time.time() - t_total)
		plt.figure()
		gen = plt.plot(epochs, accs, label='accuracy vs epoch')
		plt.legend()
		plt.show()



def run_nn(x, epoch, acc):
	"""Runs a pre-trained network. x is a flattened image of the same size as the model has been trained"""
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(acc) + ".checkpoint.meta")
		saver.restore(sess, base_dir + "/savedmodels/Alex/epoch" + str(epoch) + "acc" + "{:1.3f}".format(
			acc) + ".checkpoint")


if __name__ == '__main__':
	# freeze_support()
	train_neural_network(x)
#run_nn(x, 0, 0.01)
