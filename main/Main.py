

def main():
	from learning.AlexNet import AlexNet

	AlexNet()


if __name__ == '__main__':
	import multiprocessing

	multiprocessing.freeze_support()
	main()
