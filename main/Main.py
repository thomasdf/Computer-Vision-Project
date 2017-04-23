

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


if __name__ == '__main__':
	# import multiprocessing
	#
	# multiprocessing.freeze_support()
	main()
