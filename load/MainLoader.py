import csv
import random

import numpy as np
from PIL import Image
from PIL import ImageFile

from image.Image import Img

from load import car_path, sign_path, car_img_path, labels, sign_img_path, base_dir


def load_csv(csvpath: str):
	res = []
	with open(csvpath) as file:
		dialect = csv.Sniffer().sniff(file.read(), delimiters=';,')
		file.seek(0)
		reader = csv.reader(file, dialect=dialect)
		for row in list(reader)[1:]:
			tuple = []
			for index in row:
				tuple.append(index)
			res.append(tuple)
	return res

save_path = base_dir + '/load/loadeddata.csv'
def saveCSV(filename, entries, rownames=(("Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"))):
	with open(filename, 'w') as file:
		writer = csv.writer(file, lineterminator='\n')
		writer.writerow(rownames)
		writer.writerows(entries)

class MainLoader:

	def __init__(self, size: int, testrate:float = 0.1):
		self.size = size
		self.reset_index()
		self.data = self.load_images_csv()
		self.testindexes, self.trainindexes = self.split_data(testrate, len(self.data))
		self.test_chops = self.test_choppers()

	def reset_index(self):
		self.index_test = 0
		self.index_training = 0

	def load_images_csv(self):
		return load_csv(save_path)

	def load_images_npy(self):
		data = np.load('balanced_data_set.npy')  # [[ array([int(size*size)]) ,[int(4)] ]]

		for arr, one_hot in data:
			# arr = np.multiply(arr, 255.0)
			# Image.fromarray(arr2d, mode='L').show()

			arr2d = arr.reshape((self.size, self.size))





	def store_images(self, shuffle: bool = True):

	## load data
		car_data = load_csv(car_path)  # xmin, ymin, xmax, ymax, filename, label, url
		sign_data = load_csv(sign_path)  # Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId
	## combine data
		data = []
		for xmin, ymin, xmax, ymax, filename, label, url in car_data:
			if xmin == xmax or ymin == ymax:
				continue

			data.append((xmin, ymin, xmax, ymax, (car_img_path + filename), labels.index(label)))

		for filename, w, h, xmin, ymin, xmax, ymax, label in sign_data:
			if xmin == xmax or ymin == ymax:
				continue
			data.append((xmin, ymin, xmax, ymax, (sign_img_path + filename), 0))

	## reformat data
		labeled_data = [[], [], [], []]

		if shuffle:
			random.shuffle(data)

		for d in data:
			labeled_data[d[5]].append(d)

	## balance data

		# l0, l1, l2, l3 = labeled_data

		length = int(min((len(n) for n in labeled_data)))
		croped_data = [l[:length] for l in labeled_data]
		sum_c_data = [val for sublist in croped_data for val in sublist]

		random.shuffle(sum_c_data)

		# return croped_data

	## Open, resize, convert, save image
		the_data = []

		for x0, y0, x1, y1, f, l in sum_c_data:
			img = Image.open(f)
			ImageFile.LOAD_TRUNCATED_IMAGES = True
			img = img.convert('L')
			img = img.crop((int(x0), int(y0), int(x1), int(y1)))
			img = img.resize((self.size, self.size))
			arr = np.asarray(img).ravel()
			# arr = Img.static_normalized(arr)
			the_data.append([arr, Img.to_onehot(int(l))])

		np.save('balanced_data_set.npy', the_data) # [[ array([int(size*size)]) ,[int(4)] ]]

	def load_images(self):
		car_data = load_csv(car_path)  # xmin, ymin, xmax, ymax, filename, label, url
		sign_data = load_csv(sign_path)  # Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId

		data = []
		carrs_tresh = 62567 // 3
		carrs_num = 0
		for xmin, ymin, xmax, ymax, filename, label, url in car_data:
			if xmin == xmax or ymin == ymax:
				continue
			if label == 'Car':
				carrs_num += 1
				if carrs_num > carrs_tresh:
					continue
			data.append((xmin, ymin, xmax, ymax, (car_img_path + filename), labels.index(label)))

		for filename, w, h, xmin, ymin, xmax, ymax, label in sign_data:
			if xmin == xmax or ymin == ymax:
				continue
			data.append((xmin, ymin, xmax, ymax, (sign_img_path + filename), 0))

		saveCSV(save_path, data, ('xmin', 'ymin', 'xmax', 'ymax', 'filepath', 'label'))
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


		for i, index in enumerate(indexes):
			xmin, ymin, xmax, ymax, filepath, label = self.data[index]
			result[index] = Img.chop_coordinates(int(xmin), int(ymin), int(xmax), int(ymax), self.size, index)

		return result

	def __get_next_batch(self, batch_size: int, num_images:int, is_training: bool = True):
		if is_training:
			indexes = self.trainindexes
			croparg = lambda _: ()
			start = self.index_training
			self.index_training += num_images
			end = self.index_training

		else:
			num_images = batch_size
			indexes = self.testindexes
			croparg = lambda index: self.test_chops[index]
			start = self.index_test
			self.index_test += num_images
			end = self.index_test

		num_samples = batch_size // num_images
		batch = []
		labels = []

		assert len(indexes) > start

		if len(indexes) <= end:
			end = len(indexes)


		for i, index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = self.data[index]
			image = Image.open(filepath).convert(mode='L')
			arr2d = np.asarray(image)
			arr2d = arr2d[int(ymin):int(ymax), int(xmin):int(xmax)]
			arr2d.astype(np.float32)
			arr2d = np.multiply(arr2d, 1.0 / 255.0)

			for j in range(num_samples):
				arr_crop = Img.cropfunc(arr2d, self.size, croparg(index), is_training)
				arr1d = arr_crop.ravel()
				batch.append(arr1d)
				labels.append(Img.to_onehot(int(label)))

		# for image in batch:
		# 	assert image.shape[0] == 224*224


		stacked_batch = np.vstack(batch)
		stacked_labels = np.vstack(labels)

		return stacked_batch, stacked_labels

	def __get_test_batch(self, batch_size: int, data: [], indexes: [int], is_training: bool):
		batch = []
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
			arr_crop = Img.testcrop(arr2d, self.size, self.test_chops[index])
			arr1d = arr_crop.ravel()

			batch.append(arr1d)
			labels.append(image.one_hot)

		stacked_batch = np.vstack(batch)
		stacked_labels = np.vstack(labels)

		return stacked_batch, stacked_labels

	def __get_test_batch_queued(self, batch_size: int, data: [], indexes: [int], is_training: bool, batch_queue, label_queue):

		start = self.index_test
		self.index_test += batch_size
		end = self.index_test

		for i, index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = data[index]

			image = Img.open(filepath)
			image.crop(int(xmin), int(ymin), int(xmax), int(ymax))  # crop object
			image.convert('L')  # Convert to grayscale
			image.set_label(label)

			arr2d = image.normalized2d()
			arr_crop = Img.testcrop(arr2d, self.size, self.test_chops[index])
			arr1d = arr_crop.ravel()
			batch_queue.put(arr1d)
			label_queue.put(image.one_hot)

	def get_next_batch_unstacked(self, batch_size: int, num_images: int, is_training: bool = True):
		if is_training:
			indexes = self.trainindexes
			croparg = lambda _: ()
			start = self.index_training
			self.index_training += num_images
			end = self.index_training

		else:
			num_images = batch_size
			indexes = self.testindexes
			croparg = lambda index: self.test_chops[index]
			start = self.index_test
			self.index_test += num_images
			end = self.index_test

		num_samples = batch_size // num_images
		batch = []
		labels = []

		assert len(indexes) > start

		if len(indexes) <= end:
			end = len(indexes)

		for i, index in enumerate(indexes[start:end]):
			xmin, ymin, xmax, ymax, filepath, label = self.data[index]
			image = Image.open(filepath).convert(mode='L')
			arr2d = np.asarray(image)
			arr2d = arr2d[int(ymin):int(ymax), int(xmin):int(xmax)]
			arr2d.astype(np.float32)
			arr2d = np.multiply(arr2d, 1.0 / 255.0)

			for j in range(num_samples):
				arr_crop = Img.cropfunc(arr2d, self.size, croparg(index), is_training)
				arr1d = arr_crop.ravel()
				batch.append(arr1d)
				labels.append(Img.to_onehot(int(label)))

		# for image in batch:
		# 	assert image.shape[0] == 224*224


		# stacked_batch = np.vstack(batch)
		# stacked_labels = np.vstack(labels)

		return batch, labels


	def next_batch(self, batch_size: int, images_used: int = 1, is_training:bool = True):

		if is_training:
			return self.__get_next_batch(batch_size, images_used, is_training=True)
			# return self.__get_batch(batch_size, self.data, self.trainindexes, True)
		else:
			return self.__get_next_batch(batch_size, batch_size, is_training=False)
			# return self.__get_test_batch(batch_size, self.data, self.testindexes, False)



if __name__ == '__main__':
	ml = MainLoader(32, 0.1)
	# ml.store_images()
	ml.load_images_npy()


	# def next_batch_async(self, batch_size: int, images_used: int, is_training: bool, batch_x, batch_y, lock):
	# 	# self.__get_next_batch_queued(batch_size, images_used, is_training, batch_x, batch_y, lock)
	# 	pass
	#
	# def next_batch_async_arr(self, batch_size: int, images_used: int, is_training: bool, batch_x, batch_y,):
	# 	x, y = self.get_next_batch_unstacked(batch_size, images_used, is_training)
	# 	xx = np.concatenate(x)
	# 	yy = np.concatenate(y)
	#
	# 	xarr = np.frombuffer(batch_x.get_obj())
	# 	yarr = np.frombuffer(batch_y.get_obj())
	#
	# 	np.copyto(xarr, xx)
	# 	np.copyto(yarr, yy)

# print('Allah!')
# n = MainLoader(224, 0.1)
# print('Niqab!')
# t = time.time()
# n.next_batch(1000)
# print(time.time() - t)
# print('batchy macbatchface')
# n.next_batch(1000, False)
# print('hei')

# if __name__ == '__main__':
# 	s = 100
# 	size = 25
# 	test_s = 0.01
# 	shape = (size, size)
#
# 	ml = MainLoader(size, test_s)
# 	ml2 = MainLoader(size, test_s)
#
# 	xy = ml.next_batch(s, s, False)
# 	# ml.reset_index()
# 	xy2 = ml2.next_batch(s, s, False)
#
# 	x, y = xy
#
# 	for i in x[30:31]:
# 		a = np.multiply(i, 255.0)
# 		Image.fromarray(a.reshape(shape)).show()
#
# 	x2, y2 = xy2
# 	for i in x2[30:31]:
# 		a = np.multiply(i, 255.0)
# 		Image.fromarray(a.reshape(shape)).show()



'''
000 = {float64} 0.149019607843
001 = {float64} 0.141176470588
002 = {float64} 0.133333333333
003 = {float64} 0.129411764706
004 = {float64} 0.133333333333
005 = {float64} 0.133333333333
006 = {float64} 0.137254901961
007 = {float64} 0.137254901961

'''
