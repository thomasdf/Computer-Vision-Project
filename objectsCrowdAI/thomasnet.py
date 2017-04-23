import tensorflow as tf
import os
from tensorflow.python.client import device_lib

from load.MainLoader import MainLoader

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())
size = 32
loader = MainLoader(size ,0.05)
base_dir = os.path.dirname(os.path.dirname(__file__))

n_classes = 4
# batch size to use when loading mnist data (number of images)
batch_size = 1
num_batches = 10

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder("float", [None, size*size])  # 28x28 px images flattened
y = tf.placeholder("float")


def conv2D(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def neural_network_model(x):
	# define variables for layers (that is: allocate memory, create a structure)
	weights = {"conv1": tf.Variable(tf.random_normal([5, 5, 1, 64])),
	           "conv2": tf.Variable(tf.random_normal([5, 5, 64, 128])),
	           "fc": tf.Variable(tf.random_normal([size * size * 128, 4096])),
	           "out": tf.Variable(tf.random_normal([4096, n_classes]))
	           }

	biases = {"conv1": tf.Variable(tf.random_normal([64])),
	          "conv2": tf.Variable(tf.random_normal([128])),
	          "fc": tf.Variable(tf.random_normal([4096])),
	          "out": tf.Variable(tf.random_normal([n_classes]))
	          }

	x = tf.reshape(x, shape=[-1, size, size, 1])

	conv1 = tf.nn.relu(conv2D(x, weights["conv1"]) + biases["conv1"])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2D(conv1, weights["conv2"]) + biases["conv2"])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, int(size/4 * size/4 * 128)])
	fc = tf.nn.relu(tf.matmul(fc, weights["fc"]) + biases["fc"])

	#fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights["out"]) + biases["out"]

	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	# define cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# define optimizer (minimize cost)
	optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

	num_epochs = 50

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver()

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(num_batches):
				epoch_x, epoch_y = loader.next_batch(batch_size)  # load data from mnist dataset
				# x = image, y = class
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print("Epoch", epoch, " of ", num_epochs, " loss: ", epoch_loss)
			loader.reset_index()
			if epoch % 50 == 0:
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, "float"))
				testx, testy = loader.next_batch(batch_size, is_training=False)
				print("Accuracy: ", accuracy.eval({x: testx, y: testy}))

		saver.save(sess, base_dir + "/savedmodels/thomasnet/2k.checkpoint")


def run_neural_network(batch):
	nn_output = neural_network_model(x)

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(
			base_dir + "/savedmodels/thomasnet/2k.checkpoint.meta")
		saver.restore(sess, base_dir + "/savedmodels/thomasnet/2k.checkpoint")
		init = tf.global_variables_initializer()
		sess.run(init)
		results = sess.run(tf.nn.softmax(nn_output), feed_dict={x: batch})
		# results = sess.run(nn_output, feed_dict={x: batch})
	return results

train_neural_network(x)
#xs, ys = loader.next_batch(2, 2, False)

#for img in xs:
#	image = Img.from_array1d(img, [64,64])
#	image.denormalize()
#	image.show()
#res = run_neural_network(xs)
#print(res)
