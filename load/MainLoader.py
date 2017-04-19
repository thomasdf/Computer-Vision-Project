import os
import random

import numpy as np

from image.Image import Img


# load_images()
base_dir = os.path.dirname(os.path.dirname(__file__))

car_path = base_dir + '/datasets/object-detection-crowdai/labels.csv'
sign_path = base_dir + '/signs/csv/signs.csv'

car_img_path = base_dir + '/datasets/object-detection-crowdai/'
sign_img_path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
labels = ['signs', 'Pedestrian', 'Car', 'Truck']


class MainLoader:

	def __init__(self, size: int, testrate:float = 0.1):
		self.data = self.load_images()
		self.testindexes, self.trainindexes = self.split_data(testrate, len(self.data))
		self.index_in_epoch = 0
		self.size = size


	def load_images(self):
		from objectsCrowdAI.Loader import load_csv
		car_data = load_csv(car_path)  # xmin, ymin, xmax, ymax, filename, label, url
		sign_data = load_csv(sign_path)  # Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId

		data = []

		for xmin, ymin, xmax, ymax, filename, label, url in car_data:
			data.append((xmin, ymin, xmax, ymax, (car_img_path + filename), labels.index(label)))

		for filename, w, h, xmin, ymin, xmax, ymax, label in sign_data:
			data.append((xmin, ymin, xmax, ymax, (sign_img_path + filename), 0))

		return data  # xmin, ymin, xmax, ymax, filepath, label

	def split_data(self, testrate: float, data_length: int):
		from tools.SplitSet import hash_split
		testindexes = hash_split(testrate, data_length)
		trainindexes = list(filter(lambda x: x not in testindexes, range(data_length)))
		return testindexes, trainindexes

	def test_chop(self, data: [], indexes: [int]):
		result = {}

		for i, index in enumerate(indexes):
			xmin, ymin, xmax, ymax, filepath, label = data[index]

			pass





	def __get_batch(self, num: int, data: [], indexes: [int]):
		batch = np.zeros(num, dtype=np.ndarray)
		labels = np.zeros(num, dtype=np.ndarray)
		start = self.index_in_epoch
		self.index_in_epoch += num
		end = self.index_in_epoch

		for i,index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = data[index]

			image = Img.open(filepath)
			image.crop(int(xmin), int(ymin), int(xmax), int(ymax))  # crop object
			image.convert('L')  # Convert to grayscale
			image.set_label(label)
			batch[i] = image.normalized2d()
			labels[i] = image.one_hot
		return batch, labels



	def next_batch(self, num: int, is_training = True ):

		if is_training:
			return self.__get_batch(num, self.data, self.trainindexes)
		else:
			return self.__get_batch(num, self.data, self.testindexes)


