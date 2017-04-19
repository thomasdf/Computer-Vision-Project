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
		self.__update(image)

	def __update(self, image: PIL.Image):
		if type(image) is Img:
			raise Exception('Plis dude plis... ')

		self.image = image
		self.full_image = self.image
		self.arr2d = np.array(self.image)
		self.shape = self.arr2d.shape
		self.arr1d = self.arr2d.ravel()
		self.labels = np.zeros(4, dtype=float)
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

	def normalized(self):
		return np.array(list(map(normalize_map, self.arr1d)))
	def normalized2d(self):
		arr = self.arr2d
		arr.astype(np.float32)
		return np.multiply(arr, 1.0 / 255.0)

	def normalize(self):
		self.from_array1d(self.normalized(), self.shape, mode='L')

	def denormalize(self):
		arr = np.array(list(map(unnormalize_map, self.arr1d)))
		self.from_array1d(arr, self.shape, mode='L')

	def setLabel(self, int):
		self.labels[int] = 1

	@classmethod
	def denormalized(cls, array: np.array):
		return np.array(list(map(unnormalize_map, array)))


	def croped_arr(self, xmin, ymin, xmax, ymax):
		return self.arr2d[ymin:ymax, xmin:xmax]

	def padd(self, width, height, mode=static_mode):
		widths = np.math.ceil(width / self.shape[1])
		max_width = widths*self.shape[1]
		heights = np.math.ceil(height / self.shape[0])
		max_height = heights*self.shape[0]

		new_img = Image.new(mode, (max_width, max_height))

		x_offset = 0
		y_offset = 0
		for _ in range(heights):
			for __ in range(widths):
				new_img.paste(self.image, (x_offset, y_offset))
				x_offset += self.shape[1]
			y_offset += self.shape[0]
			x_offset = 0

		return np.array(new_img.crop((0, 0, height, width)))

	@classmethod
	def to_test_crop(cls, xmin, ymin, xmax, ymax, size, seed):
		height = xmax - xmin
		width = ymax - ymin

		size1d = (height-size)*(width-size)

		pos1d = seed % size1d
		x = pos1d % width
		y = pos1d // width

		return x, y, x + size, y + size




	def rand_crop(self, height, width):
		if width >= self.shape[1] or height >= self.shape[0]:
			return self.padd(width, height)



		rand_gen = lambda r: random.randint(0, r)
		rand_y = rand_gen(self.shape[0])
		rand_x = rand_gen(self.shape[1])

		offset_y = sign() * width + rand_y
		offset_x = sign() * height + rand_x

		if not(0 < offset_y < self.shape[0]):
			self.rand_crop(width, height)
		if not(0 < offset_x < self.shape[1]):
			self.rand_crop(width, height)

		x = (rand_x, offset_x)
		y = (rand_y, offset_y)


		return self.croped_arr(min(x), min(y), max(x), max(y))





