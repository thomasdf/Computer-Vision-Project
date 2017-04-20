import numpy as np
import tensorflow as tf
import os

import time
from tensorflow.python.client import device_lib

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from load.MainLoader import MainLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

#batch size to use when loading data
test_set_size = 0.1
batch_size = 500
num_epochs = 1
n_classes = 4
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

loader = MainLoader(28, test_set_size)
print("loader initialized")

x = tf.placeholder("float", [None, 28*28])
#x = tf.placeholder("float", None)
y = tf.placeholder("float")

def conv2D(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = "SAME")

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def neural_network_model(x):

	#define variables for layers (that is: allocate memory, create a structure)
	weights = {"conv1": tf.Variable(tf.random_normal([5,5,1,64])),
			   "conv2": tf.Variable(tf.random_normal([5,5,64,128])),
			   "fc": tf.Variable(tf.random_normal([7*7*128, 1024])),
			   "out": tf.Variable(tf.random_normal([1024, n_classes]))
			   }


	biases = {"conv1": tf.Variable(tf.random_normal([64])),
			   "conv2": tf.Variable(tf.random_normal([128])),
			   "fc": tf.Variable(tf.random_normal([1024])),
			   "out": tf.Variable(tf.random_normal([n_classes]))
			   }

	x = tf.reshape(x, shape=[-1,28,28,1])

	conv1 = tf.nn.relu(conv2D(x, weights["conv1"]) + biases["conv1"])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2D(conv1, weights["conv2"]) + biases["conv2"])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 7*7*128])
	fc = tf.nn.relu(tf.matmul(fc, weights["fc"]) + biases["fc"])

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights["out"]) + biases["out"]

	return output

def train_neural_network(x):
	print("start neural network training")
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels=y) )

	#define optimizer (minimize cost)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		num_batches = int(np.ceil((len(loader.data)*(1-test_set_size))/batch_size))

		for epoch in range(num_epochs):
			epoch_loss = 0
			t0 = time.time()
			for batch_num in range(num_batches):
				t_batch = time.time()
				epoch_x, epoch_y = loader.next_batch(batch_size) #load data from mnist dataset
				print('\tbatch time', time.time() - t_batch)


				#x = image, y = class
				batch, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
				t1 = time.time()
				print("Batch ", batch_num, " of ", num_batches, " complete. Loss ", epoch_loss, ' batch', batch, ' time', (t1 - t0))
				t0 = t1
			print("Epoch", epoch, " of ", num_epochs, " loss: ", epoch_loss)

	   # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

	   # accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	   # print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
