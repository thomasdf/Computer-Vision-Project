
import tensorflow as tf

from load.MainLoader import MainLoader

tf.set_random_seed(0)

size = 28
length = size*size
labelsize = 4

X = tf.placeholder(tf.float32, [None, size, size, 1])
W = tf.Variable(tf.zeros([length, labelsize]))
b = tf.Variable(tf.zeros([labelsize]))

init = tf.initialize_all_variables()

# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, length]), W) + b)

# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, labelsize])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

loader = MainLoader(15, 0.1)
epoch = 0
for i in range(1000):
	# load batch of images and correct answers
	train_batch_X, train_batch_Y = loader.next_batch(100, is_training=True)
	train_data = {X: train_batch_X, Y_: train_batch_Y}

	# train
	sess.run(train_step, feed_dict=train_data)
	if i % 10 == 9:
		# success ?
		a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
		print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
	if i % 100 == 99:

		test_batch_X, test_batch_Y = loader.next_batch(100, is_training=False)
		# success on test data ?
		test_data = {X: test_batch_X, Y_: test_batch_Y}
		a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
		print(str(i) + ": ********* epoch " + str(epoch) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
		epoch += 1
