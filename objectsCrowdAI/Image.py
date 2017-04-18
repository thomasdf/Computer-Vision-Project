import numpy as np
from PIL import Image


class Img:
	def __init__(self, path):
		self.image = Image.open(path)
		self.arr2d = np.array(self.image)
		self.shape = self.arr2d.shape
		self.arr1d = self.arr2d.ravel()

	def show(self):
		self.image.show()

	def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
		self.croped = self.image.crop((xmin, ymin, xmax, ymax))
		return self.croped

