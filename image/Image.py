import random

import PIL
import numpy as np
from PIL import Image

range_map = lambda input_start, input_end, output_start, output_end: \
	lambda input: output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)

normalize_map = range_map(0, 255, 0, 1)
unnormalize_map = lambda x: int(round(range_map(0, 1, 0, 255)(x)))

static_mode = 'L'
static_num_labels = 4

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
	def static_normalized2d(cls, array: np.ndarray):
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
		widths = int(np.ceil(sample_img_size / array.shape[1]))
		heights = int(np.ceil(sample_img_size / array.shape[0]))
		wide_arr = np.concatenate([array for _ in range(widths)], axis=0)
		full_arr = np.concatenate([wide_arr for _ in range(heights)], axis=1)

		return full_arr[0:sample_img_size, 0:sample_img_size]

	@classmethod
	def chop_coordinates(cls, xmin:int, ymin:int, xmax:int, ymax:int, sample_img_size:int, seed:int):
		height = xmax - xmin
		width = ymax - ymin
		# if width == 0:
		# 	return None

		size1d = (height - sample_img_size) * (width - sample_img_size)
		pos1d = 0 if size1d == 0 else seed % size1d
		x = pos1d % width
		y = pos1d // width

		return x, y, x + sample_img_size, y + sample_img_size



	def get_train_arr1d(self, size: int):
		arr2d = self.randcrop(self.arr2d, size)
		arr2d = Img.static_normalized2d(arr2d)
		return arr2d.ravel()

	def get_test_arr1d(self):
		pass

	@classmethod
	def randcrop(cls, array: np.ndarray, sampel_img_size: int):
		if sampel_img_size >= array.shape[1] or sampel_img_size >= array.shape[0]:
			return cls.padd(array, sampel_img_size)

		return cls.croparray(array, *cls.chop_coordinates(0, 0, array.shape[1], array.shape[0], sampel_img_size, random.randint(0, 4000013)))

	def rand_crop(self, height, width):
		if width >= self.shape[1] or height >= self.shape[0]:
			return self.padd(width, height)

		rand_gen = lambda r: random.randint(0, r)

		while True:
			rand_y = rand_gen(self.shape[0])
			rand_x = rand_gen(self.shape[1])
			offset_y = sign() * width + rand_y
			offset_x = sign() * height + rand_x
			if (0 <= offset_y < self.shape[0]) and (0 <= offset_x < self.shape[1]):
				break




		x = (rand_x, offset_x)
		y = (rand_y, offset_y)


		return self.croparray(min(x), min(y), max(x), max(y))





