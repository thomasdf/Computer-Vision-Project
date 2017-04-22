import numpy as np
import tensorflow as tf
import os

import time
from tensorflow.python.client import device_lib
from load.MainLoader import labels

from tensorflow.examples.tutorials.mnist import input_data

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from load.MainLoader import MainLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

###########################################################################################
###                                 THINGS                                              ###
###########################################################################################

# data loader
test_set_size = 0.1  # fraction of dataset used as test-set
loader = MainLoader(224, test_set_size)
print("loader initialized")

# data things
batch_size = 50
num_batches = int(np.ceil((len(loader.data) * (1 - test_set_size)) / batch_size))

# training things
num_epochs = 1
dropout_rate = 0.2
lr = 0.00001

# classifier things
size = 224 # (X * X size)
n_classes = len(labels)

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

	print(tf.shape(pool3))

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
			for batch_num in range(0):
				t_batch = time.time()
				batch_x, batch_y = loader.next_batch(batch_size)  # load data from mnist dataset
	#			batch_x, batch_y = mnist.train.next_batch(batch_size)  # load data from mnist dataset
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
				test_batch_x, test_batch_y = loader.next_batch(batch_size, is_training=False)
				#test_batch_x, test_batch_y = mnist.test.next_batch(batch_size)
				epoch_acc += accuracy.eval({x: test_batch_x, y: test_batch_y})
				print("Calculating accuracy. ", "{:10.2f}".format((_ / num_batches) * 100), "% complete.")
			print("Epoch Accuracy: ", epoch_acc / num_batches)


train_neural_network(x)
