import os

import time

import numpy as np
import tensorflow as tf
from PIL import Image

from learning.SlidingWindowV2 import slide, shade2d
from main.alexnet import alexnet
from tools import ArrayTool

base_dir = os.path.dirname(os.path.dirname(__file__)) + '/'
random_pic_path = base_dir + 'datasets/object-detection-crowdai/1479498442970431250.jpg'

def train_deeper():
	from learning.deepernet import DeeperNet
	classifier = DeeperNet()
	classifier.train_neural_network(None)


def main():
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	# from learning.AlexNet import AlexNet
	from learning.thomasnetv2 import ThomasNet
	# from learning.deepernet import DeeperNet
	# classifier = DeeperNet()
	# classifier.init_loader()
	# # classifier.train_neural_network()
	#
	# a = ArrayTool.out(Image.open(random_pic_path), classifier, 9, 0.582, .25, True)
	# # Image.fromarray(a).show()

	# classifier.train_neural_network()

	size = 32
	LR = 1e-3
	EPOCHS = 10

	model = alexnet(size, size, LR, 4)
	hard = 'models/cvp-0.001-ravnanetv2-10-epochs-data.model'
	MODEL_NAME = hard
		# 'cvp-{}-{}-{}-epochs-data.model'.format(LR, 'ravnanet', EPOCHS)
	model.load(MODEL_NAME)

	img = Image.open(random_pic_path)

	a = new_out(img, size, model, 0.30, False)
	Image.fromarray(a).show()


def test():
	size = 32
	LR = 1e-3
	EPOCHS = 10
	# MODEL_NAME = base_dir + '/main/log/cvp-{}-{}-{}-epochs-300K-data.model'.format(LR, 'ravnanet', EPOCHS)
	model = alexnet(size, size, LR, 4)
	# model.load(MODEL_NAME)
	img = Image.open(random_pic_path)

	saver = tf.train.Saver()

	chachpoint_path = 'C:/Users/kiwi/IdeaProjects/Computer-Vision-Project/main/checkpoint'
	x = tf.placeholder("float", [None, size, size])
	with tf.Session() as sess:

		saver.restore(sess=sess, save_path=chachpoint_path)
		c, batches = slicy(img, size)
		# for batch in batches:
		res = sess.run(model, feed_dict={x: batches})

		a = shady(img, size, c, res)


	# a = new_out(img, size,  model,  9, 0.582, .25, True)
	Image.fromarray(a).show()

def slicy(img: Image, size: int):
	arr = np.asarray(img.convert('L'))

	return slide(arr, size, size)

def shady(img: Image, size: int, coor, predictions, treshold: float = .70, scaled_shade: bool = True):
	c = zip(coor, predictions)
	return shade2d(img, c, size, 255, treshold, scaled_shade)


def new_out(img: Image, size: int, model, treshold: float = .70, scaled_shade: bool = True):
	ttot = time.time()
	#
	# clsfy = np.vectorize(classifier)

	arr = np.asarray(img.convert('L'))

	tslide = time.time()
	coor, slices = slide(arr, size, size)
	tslide = time.time() - tslide
	tclasfy = time.time()
	# c = []
	# for x, y, e in b:
	#     c.append((x, y, classifier(e)))
	# r = classifier.run_nn(slices, epoch, acc)
	# r = []
	r = model.predict(slices.reshape(-1, size, size, 1))

	# for s in slices:
	#
	# 	prediction = q[0]
	# 	r.append(prediction)

	tclasfy = time.time() - tclasfy
	c = zip(coor, r)
	a = shade2d(img, c, size, 160, treshold, scaled_shade)

	tshade = time.time()
	tshade = time.time() - tshade

	ttot = time.time() - ttot

	print('slide', tslide)
	print('classify', tclasfy)
	print('shade', tshade)
	print('tot', ttot)
	return a


if __name__ == '__main__':
	# import multiprocessing
	#
	# multiprocessing.freeze_support()
	# train_deeper()
	main()
