import numpy as np
import tensorflow as tf
import os

import time
from tensorflow.python.client import device_lib
from load.MainLoader import labels

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from load.MainLoader import MainLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

###########################################################################################
###                                 THINGS                                              ###
###########################################################################################

# data loader
test_set_size = 0.1  # fraction of dataset used as test-set
loader = MainLoader(28, test_set_size)
print("loader initialized")

# data things
batch_size = 500
num_batches = int(np.ceil((len(loader.data) * (1 - test_set_size)) / batch_size))

# training things
num_epochs = 1
dropout_rate = 0.2
lr = 0.00001

# classifier things
size = 28  # (X * X size)
n_classes = len(labels)

# tensorflow things
x = tf.placeholder("float", [None, size * size])
y = tf.placeholder("float")


###########################################################################################
###                                 TF/NN                                               ###
###########################################################################################

def neural_network_model(x, is_training: bool = True):
	"""Defines the neural network model. Output is a n_labels long array"""

	# input layer
	input = tf.reshape(x, shape=[-1, 28, 28, 1])

	# convolutional layer 1
	conv1 = tf.layers.conv2d(
		inputs=input,
		filters=32,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu,
	)  # [batchsize, 28, 28, 32]

	# max-pooling layer 1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=2,  # output of: [batchsize, input/2, input/2, same as conv1 (e.g. 32)]
		strides=2,  # input/2 because we max pool 2x2 pixels with a stride of 2 (that is: no overlap)
	)  # [batchsize, 14, 14, 32]

	# convolutional layer 2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,  # output: [batchsize, input, input, 64]
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
	pool2flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

	# fully connected layer:
	fc = tf.layers.dense(
		inputs=pool2flattened,
		units=1024,
		activation=tf.nn.relu,
	)

	# add dropout
	dropout = tf.layers.dropout(
		inputs=fc,
		rate=dropout_rate,
		training=is_training,
	)  # [batchsize, 1024]

	output = tf.layers.dense(
		inputs=dropout,
		units=n_classes,
	)

	return output


def train_neural_network(x):
	print("start neural network training")

	nn_output = neural_network_model(x)

	# cost function
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=y))

	# optimizer function
	optimizer_func = tf.train.GradientDescentOptimizer(lr).minimize(cost_func)

	# init variables and session
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost = 0
			t0 = time.time()
			for batch_num in range(num_batches):
				t_batch = time.time()
				batch_x, batch_y = loader.next_batch(batch_size)  # load data from mnist dataset
				print('\tbatch loading time', time.time() - t_batch)

				batch, c = sess.run([optimizer_func, cost_func], feed_dict={x: batch_x, y: batch_y})
				epoch_cost += c
				t1 = time.time()
				print("Batch ", batch_num, " of ", num_batches, " complete. Loss ", epoch_cost, ' batch', batch,
				      'training time', (t1 - t0))
				t0 = t1
			print("Epoch", epoch, " of ", num_epochs, " loss: ", epoch_cost)

			correct = tf.equal(tf.argmax(nn_output, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))

			print("Epoch complete. Calculating accuracy...")

			epoch_acc = 0
			for _ in range(num_batches):
				test_batch_x, test_batch_y = loader.next_batch(batch_size, False)
				epoch_acc += accuracy.eval({x: test_batch_x, y: test_batch_y})
				print("Calculating accuracy. ", "{:10.2f}".format((_ / num_batches) * 100), "% complete.")
			print("Epoch Accuracy: ", epoch_acc / num_batches)


train_neural_network(x)
