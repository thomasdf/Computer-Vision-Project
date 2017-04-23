

def main():
	from learning.AlexNet import AlexNet
	from learning.thomasnetv2 import ThomasNet
	tn = ThomasNet()
	# tn.train_neural_network()
	tn.run_nn()
	#AlexNet()


if __name__ == '__main__':
	# import multiprocessing
	#
	# multiprocessing.freeze_support()
	main()
