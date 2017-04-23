import ctypes
import multiprocessing

import numpy as np

from load import MainLoader


class LoadProcess():

	def __init__(self, loader: MainLoader, batch_size: int, image_load_size: int, size:int, n_classes: int):
		multiprocessing.freeze_support()
		self.target = loader.next_batch_async_arr
		self.batch_size = batch_size
		self.image_load_size = image_load_size
		self.flat_batch_size = size * size * batch_size
		self.flat_labels_size = batch_size * n_classes
		self.batch_shape = (batch_size, size*size)
		self.labels_shape = (batch_size, n_classes)
		self.x_arr_batch = multiprocessing.Array(ctypes.c_double, self.flat_batch_size)
		self.y_arr_batch = multiprocessing.Array(ctypes.c_double, self.flat_labels_size)

	def setup(self, target: callable, args):
		self.target = target
		self.args = args

	def loadunits(self, batch_x, batch_y):
		self.batch_x, self.batch_y = batch_x, batch_y

	def runtraining(self):
		# self.target(self.batch_size, self.image_load_size, True, self.x_arr_batch, self.y_arr_batch)
		self.process = multiprocessing.Process(target=self.target,
		                                       args=(self.batch_size, self.image_load_size, True, self.x_arr_batch, self.y_arr_batch))
		self.process.start()

	def get_batch(self):
		x_b = np.frombuffer(self.x_arr_batch.get_obj()).reshape(self.batch_shape)
		y_b = np.frombuffer(self.y_arr_batch.get_obj()).reshape(self.labels_shape)
		return x_b, y_b

	def wait(self):
		self.process.join()

