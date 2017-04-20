import os
import random

import numpy as np

from image.Image import Img


# load_images()
from tools.SplitSet import hash_split

base_dir = os.path.dirname(os.path.dirname(__file__))

car_path = base_dir + '/datasets/object-detection-crowdai/labels.csv'
sign_path = base_dir + '/signs/csv/signs.csv'

car_img_path = base_dir + '/datasets/object-detection-crowdai/'
sign_img_path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
labels = ['signs', 'Pedestrian', 'Car', 'Truck']


class MainLoader:

	def __init__(self, size: int, testrate:float = 0.1):
		self.size = size
		self.index_in_epoch = 0
		self.data = self.load_images()
		self.testindexes, self.trainindexes = self.split_data(testrate, len(self.data))
		self.test_chops = self.test_choppers()



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
		# path = base_dir + '/load/testindexes.txt'
		# with open(path, 'w') as file:
		# 	for i in testindexes:
		# 		file.write(str(i) + ',')


		trainindexes = list(filter(lambda x: x not in testindexes, range(data_length)))
		return testindexes, trainindexes

	def test_choppers(self, indexes: [int] = None):
		result = {}
		if indexes == None:
			indexes = self.testindexes
		# seeds = hash_split(1, len(indexes))
		# path = base_dir + '/load/seeds.txt'
		# with open(path, 'w') as file:
		# 	for i in seeds:
		# 		file.write(str(i) + ',')

		print('End hash')
		for i, index in enumerate(indexes):
			xmin, ymin, xmax, ymax, filepath, label = self.data[index]
			result[index] = Img.to_test_crop(int(xmin), int(ymin), int(xmax), int(ymax), self.size, index)

		return result





	def __get_batch(self, num: int, data: [], indexes: [int], is_testing: bool):
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
			arr = image.normalized2d()
			xmin, ymin, xmax, ymax = self.test_chops[index] # min:int, ymin:int, xmax:int, ymax:int
			n_arr = arr[ymin:ymax, xmin:xmax]

			batch[i] = n_arr
			labels[i] = image.one_hot
		return batch, labels



	def next_batch(self, num: int, is_training:bool = True):

		if is_training:
			return self.__get_batch(num, self.data, self.trainindexes, True)
		else:
			return self.__get_batch(num, self.data, self.testindexes, False)



n = MainLoader(15, 0.1)
# n.test_choppers()

print('hei')
