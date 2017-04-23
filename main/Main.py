import os

from PIL import Image

from tools import ArrayTool

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'


def main():
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	from learning.AlexNet import AlexNet
	from learning.thomasnetv2 import ThomasNet
	from learning.deepernet import DeeperNet
	classifier = DeeperNet()
	classifier.train_neural_network()
	#ThomasNet()
	#AlexNet()

	a = ArrayTool.out(Image.open(random_pic_path), ThomasNet(), 3, 0.552)
	Image.fromarray(a).show()

if __name__ == '__main__':
	# import multiprocessing
	#
	# multiprocessing.freeze_support()
	main()
