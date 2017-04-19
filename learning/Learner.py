
import tensorflow as tf

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

for i in range(1000):
	# load batch of images and correct answers
	batch_X, batch_Y = mnist.train.next_batch(100)
	train_data = {X: batch_X, Y_: batch_Y}

	# train
	sess.run(train_step, feed_dict=train_data)

	# success ?
	a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

	# success on test data ?
	test_data = {X: mnist.test.images, Y_: mnist.test.labels}


	a, c = sess.run([accuracy, cross_entropy], feed=test_data)
