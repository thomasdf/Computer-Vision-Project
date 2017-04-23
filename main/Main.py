import os

from PIL import Image

from tools import ArrayTool

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

def train_deeper():
	from learning.deepernet import DeeperNet
	classifier = DeeperNet()
	classifier.train_neural_network(None)


def main():
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	from learning.AlexNet import AlexNet
	from learning.thomasnetv2 import ThomasNet
	from learning.deepernet import DeeperNet
	classifier = DeeperNet()
	# classifier.train_neural_network()

	a = ArrayTool.out(Image.open(random_pic_path), classifier, 9, 0.582, -1)
	Image.fromarray(a).show()


if __name__ == '__main__':
	# import multiprocessing
	#
	# multiprocessing.freeze_support()
	#train_deeper()
	main()
