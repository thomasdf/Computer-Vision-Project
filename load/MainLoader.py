import os
import random

import numpy as np
from PIL import Image

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
		self.reset_index()
		self.data = self.load_images()
		self.testindexes, self.trainindexes = self.split_data(testrate, len(self.data))
		self.test_chops = self.test_choppers()

	def reset_index(self):
		self.index_test = 0
		self.index_training = 0

	def load_images(self):
		from objectsCrowdAI.Loader import load_csv
		car_data = load_csv(car_path)  # xmin, ymin, xmax, ymax, filename, label, url
		sign_data = load_csv(sign_path)  # Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId

		data = []

		for xmin, ymin, xmax, ymax, filename, label, url in car_data:
			if xmin == xmax or ymin == ymax:
				continue

			data.append((xmin, ymin, xmax, ymax, (car_img_path + filename), labels.index(label)))

		for filename, w, h, xmin, ymin, xmax, ymax, label in sign_data:
			if xmin == xmax or ymin == ymax:
				continue
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

		for i, index in enumerate(indexes):
			xmin, ymin, xmax, ymax, filepath, label = self.data[index]
			result[index] = Img.chop_coordinates(int(xmin), int(ymin), int(xmax), int(ymax), self.size, index)

		return result

	def __get_training_batch(self, batch_size: int, num_images: int):
		data = self.data
		indexes = self.trainindexes
		num_samples = batch_size // num_images

		batch = []
		labels = []
		start = self.index_training
		self.index_training += num_images
		end = self.index_training

		for i, index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = data[index]

			# image = Img.open(filepath, mode='L')
			image = Image.open(filepath).convert(mode='L')

			# image = image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
			arr2d = np.array(image)
			arr2d = arr2d[int(ymin):int(ymax), int(xmin):int(xmax)]


			arr2d.astype(np.float32)
			arr2d = np.multiply(arr2d, 1.0 / 255.0)

			# image.convert('L')  # Convert to grayscale
			# img_arr = image.normalized2d()

			# arr2d = image.rand_crop(self.size, self.size)
			# arr2d = Img.static_normalized2d(arr2d)

			for j in range(num_samples):
				arr2d = Img.randcrop(arr2d, self.size)
				batch.append(arr2d.ravel())
				labels.append(label)

		stacked_batch = np.vstack(batch)
		stacked_labels = np.vstack(labels)

		return stacked_batch, stacked_labels

	def __get_test_batch(self, batch_size: int, data: [], indexes: [int], is_training: bool):
		# batch = np.ndarray(num, dtype=np.ndarray)
		batch = []
		# labels = np.ndarray(num, dtype=np.ndarray)
		labels = []
		start = self.index_test
		self.index_test += batch_size
		end = self.index_test

		for i,index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = data[index]

			image = Img.open(filepath)
			image.crop(int(xmin), int(ymin), int(xmax), int(ymax))  # crop object
			image.convert('L')  # Convert to grayscale
			image.set_label(label)

			arr2d = image.normalized2d()
			xmin, ymin, xmax, ymax = self.test_chops[index]
			arr2d = arr2d[ymin:ymax, xmin:xmax]
			arr1d = arr2d.ravel()

			batch.append(arr1d)
			labels.append(image.one_hot)

		stacked_batch = np.vstack(batch)
		stacked_labels = np.vstack(labels)

		return stacked_batch, stacked_labels



	def next_batch(self, batch_size: int, is_training:bool = True):

		if is_training:
			return self.__get_training_batch(batch_size, batch_size//10)
			# return self.__get_batch(batch_size, self.data, self.trainindexes, True)
		else:
			return self.__get_test_batch(batch_size, self.data, self.testindexes, False)



# n = MainLoader(15, 0.1)
# # n.test_choppers()
#
# print('hei')
