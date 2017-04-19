import os
import random

from image.Image import Img
from tools.SplitSet import hashSplit


# load_images()
base_dir = os.path.dirname(os.path.dirname(__file__))

car_path = base_dir + '/datasets/object-detection-crowdai/labels.csv'
sign_path = base_dir + '/signs/csv/signs.csv'

car_img_path = base_dir + '/datasets/object-detection-crowdai/'
sign_img_path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
labels = ['signs', 'Pedestrian', 'Car', 'Truck']


class MainLoader:

	def __init__(self):
		self.data = self.load_images()
		self.testindexes, self.trainindexes = self.split_data(0.1, len(self.data))

	def load_images(self):
		from objectsCrowdAI.Loader import loadCSV
		car_data = loadCSV(car_path)  # xmin, ymin, xmax, ymax, filename, label, url
		sign_data = loadCSV(sign_path)  # Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId

		data = []

		for xmin, ymin, xmax, ymax, filename, label, url in car_data:
			data.append((xmin, ymin, xmax, ymax, (car_img_path + filename), labels.index(label)))

		for filename, w, h, xmin, ymin, xmax, ymax, label in sign_data:
			data.append((xmin, ymin, xmax, ymax, (sign_img_path + filename), 0))

		return data  # xmin, ymin, xmax, ymax, filepath, label

	def split_data(self, testrate: float, data_length: int):
		testindexes = hashSplit(testrate, data_length)
		trainindexes = list(filter(lambda x: x not in testindexes, range(data_length)))
		return testindexes, trainindexes

	def get_batch(self, num: int, data: [], indexes: [int]):
		batch = []
		labels = []
		for _ in range(num):
			xmin, ymin, xmax, ymax, filepath, label = data[indexes[random.randint(0, len(indexes))]]

			image = Img.open(filepath)
			image.crop(int(xmin), int(ymin), int(xmax), int(ymax))  # crop object
			image.convert('L')  # Convert to grayscale
			image.set_label(label)

			batch.append(image.arr2d)
			labels.append(image.one_hot)

		return batch, labels



	def next_batch(self, num: int, is_training = True ):

		if is_training:
			return self.get_batch(num, self.data, self.trainindexes)
		else:
			return self.get_batch(num, self.data, self.testindexes)


