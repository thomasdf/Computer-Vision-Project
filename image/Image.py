import random

import PIL
import numpy as np
from PIL import Image

from image import static_num_labels, static_mode, unnormalize_map


def sign():
	sign = random.randint(0, 1)
	return -1 if sign is 0 else sign

class Img:

	def __init__(self, image:PIL.Image):
		self.labels = np.zeros(static_num_labels, dtype=float)
		self.full_image = image
		self.__update(image)

	def __update(self, image: PIL.Image):
		if type(image) is Img:
			raise Exception('Plis dude plis... ')

		self.image = image
		self.arr2d = np.array(self.image)
		self.shape = self.arr2d.shape
		self.arr1d = self.arr2d.ravel()
		return self


	@classmethod
	def from_array2d(cls, array2d: np.array, mode=static_mode):
		return cls(Image.fromarray(array2d, mode))

	@classmethod
	def from_array1d(cls, array1d: np.ndarray, shape: [], mode=static_mode, dtype=np.uint8):
		return cls.from_array2d(np.asarray(array1d, dtype=dtype).reshape(shape), mode)

	@classmethod
	def from_image(cls, image:PIL.Image):
		return cls(image)

	@classmethod
	def open(cls, path: str, mode=static_mode):
		return cls(Image.open(path))

	def set_label(self, labelID:int):
		if labelID >= static_num_labels:
			raise Exception('Nooooooo!')

		self.label = labelID
		self.one_hot = np.zeros(static_num_labels)
		self.one_hot[labelID] = 1

	@classmethod
	def to_onehot(cls, label:int):
		one_hot = np.zeros(static_num_labels)
		one_hot[label] = 1
		return one_hot

	def show(self):
		self.image.show()

	def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
		return self.__update(self.croped(xmin, ymin, xmax, ymax))

	def croped(self, xmin: int, ymin: int, xmax: int, ymax: int):
		return self.image.crop((xmin, ymin, xmax, ymax))

	def update(self, mode=static_mode):
		return self.__update(Image.fromarray(self.arr2d, mode))

	def converted(self, mode):
		return self.image.convert(mode)

	def convert(self, mode):
		return self.__update(self.converted(mode))

	def normalized1d(self):
		arr = self.arr1d
		arr.astype(np.float32)
		return np.multiply(arr, 1.0 / 255.0)

	def normalized2d(self):
		arr = self.arr2d
		arr.astype(np.float32)
		return np.multiply(arr, 1.0 / 255.0)

	@classmethod
	def static_normalized(cls, array: np.ndarray):
		array.astype(np.float32)
		return np.multiply(array, 1.0 / 255.0)

	def normalize(self):
		self.__update(Image.fromarray(self.normalized2d(), mode='L'))

	def denormalize(self):
		arr = self.arr2d
		arr.astype(np.float32)
		arr = np.multiply(arr, 255.0)

		self.__update(Image.fromarray(arr, mode='L'))

	def setLabel(self, int):
		self.labels[int] = 1

	@classmethod
	def denormalized(cls, array: np.array):
		return np.array(list(map(unnormalize_map, array)))

	@classmethod
	def croparray(self, array2d: np.ndarray, xmin, ymin, xmax, ymax):
		return array2d[ymin:ymax, xmin:xmax]




	@classmethod
	def padd(cls, array:np.ndarray, sample_img_size):
		widths = int(np.ceil(sample_img_size / array.shape[0]))
		heights = int(np.ceil(sample_img_size / array.shape[1]))
		wide_arr = np.concatenate([array for _ in range(widths)], axis=0)
		full_arr = np.concatenate([wide_arr for _ in range(heights)], axis=1)

		return full_arr[0:sample_img_size, 0:sample_img_size]

	@classmethod
	def chop_coordinates(cls, xmin:int, ymin:int, xmax:int, ymax:int, sample_img_size:int, seed:int):
		width = xmax - xmin
		height = ymax - ymin
		if height <= sample_img_size or width <= sample_img_size :
			return xmin, ymin, xmax, ymax

		# if width == 0:
		# 	return None
		new_width = width - sample_img_size
		size1d = (height - sample_img_size) * new_width
		pos1d = seed % size1d if size1d != 0 else 0
		x =  (pos1d % new_width)
		y =  (pos1d // new_width)

		return x, y, x + sample_img_size, y + sample_img_size

	@classmethod
	def randcrop(cls, array: np.ndarray, sampel_img_size: int):
		if array.shape[0] < sampel_img_size or array.shape[1] < sampel_img_size:
			return cls.padd(array, sampel_img_size)

		xmin, ymin, xmax, ymax = cls.chop_coordinates(0, 0, array.shape[1], array.shape[0], sampel_img_size, random.randint(0, 4000013))

		if xmax - xmax <= sampel_img_size or ymax - ymin <= sampel_img_size:
			return cls.padd(array, sampel_img_size)

		return cls.croparray(array, xmin, ymin, xmax, ymax)

	@classmethod
	def testcrop(cls, array: np.ndarray, sampel_img_size: int, chop_coordinates: [int]):
		if array.shape[0] < sampel_img_size or array.shape[1] < sampel_img_size:
			return cls.padd(array, sampel_img_size)

		xmin, ymin, xmax, ymax = chop_coordinates

		if xmax - xmax <= sampel_img_size or ymax - ymin <= sampel_img_size:
			return cls.padd(array, sampel_img_size)

		x = xmax - xmin
		y = ymax - ymin

		return cls.croparray(array, 0, 0, x, y)

	@classmethod
	def cropfunc(cls, array: np.ndarray, sampel_img_size: int, chop_coordinates: [int], is_training: bool):

		if is_training:
			arr = cls.randcrop(array, sampel_img_size)

		else:
			arr = cls.testcrop(array, sampel_img_size, chop_coordinates)

		return arr[0:sampel_img_size, 0:sampel_img_size]
