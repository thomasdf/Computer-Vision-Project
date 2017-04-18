import numpy as np
from PIL import Image


class Img:

	def __init__(self, image:Image):
		self.image = image
		self.full_image = self.image
		self.arr2d = np.array(self.image)
		self.shape = self.arr2d.shape
		self.arr1d = self.arr2d.ravel()

	@classmethod
	def from_array2d(cls, array2d: np.array, mode='RGB'):
		return cls(Image.fromarray(array2d, mode))

	@classmethod
	def from_array1d(cls, array1d: np.ndarray, shape: list, mode='RGB'):
		return cls.from_array2d(np.asarray(array1d).reshape(shape))

	@classmethod
	def from_image(cls, image:Image):
		return cls(image)

	@classmethod
	def open(cls, path: str):
		cls(Image.open(path))


	def show(self):
		self.image.show()

	def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
		self.image = self.image.crop((xmin, ymin, xmax, ymax))
		return self

	def update(self, mode='RGB'):
		self.image = Image.fromarray(self.arr2d, mode)

	def convert(self, mode):
		return self.image.convert(mode)
