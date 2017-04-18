import PIL
import numpy as np
from PIL import Image

range_map = lambda input_start, input_end, output_start, output_end: \
	lambda input: output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)

normalize_map = range_map(0, 255, 0, 1)
unnormalize_map = lambda x: int(round(range_map(0, 1, 0, 255)(x)))



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
		self.label = np.zeros(46)
		return self

	@classmethod
	def from_array2d(cls, array2d: np.array, mode='RGB'):
		return cls(Image.fromarray(array2d, mode))

	@classmethod
	def from_array1d(cls, array1d: np.ndarray, shape: [], mode='RGB', dtype=np.uint8):
		return cls.from_array2d(np.asarray(array1d, dtype=dtype).reshape(shape), mode)

	@classmethod
	def from_image(cls, image:PIL.Image):
		return cls(image)

	@classmethod
	def open(cls, path: str, mode='L'):
		return cls(Image.open(path))


	def show(self):
		self.image.show()

	def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
		return self.__update(self.image.crop((xmin, ymin, xmax, ymax)))

	def update(self, mode='RGB'):
		return self.__update(Image.fromarray(self.arr2d, mode))

	def converted(self, mode):
		return self.image.convert(mode)

	def convert(self, mode):
		return self.__update(self.image.convert(mode))

	def normalized(self):
		return np.array(list(map(normalize_map, self.arr1d)))

	def normalize(self):
		arr = np.array(list(map(normalize_map, self.arr1d)))
		self.from_array1d(arr, self.shape, mode='L')

	def denormalize(self):
		arr = np.array(list(map(unnormalize_map, self.arr1d)))
		self.from_array1d(arr, self.shape, mode='L')

	@classmethod
	def denormalized(cls, array: np.array):
		return np.array(list(map(unnormalize_map, array)))
