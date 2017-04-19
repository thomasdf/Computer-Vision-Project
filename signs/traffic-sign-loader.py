import csv
import random

import numpy as np
from PIL import Image

from image.Image import Img
from objectsCrowdAI.Loader import base_dir, load_csv
from tools.SplitSet import hash_split

path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
save_path = base_dir + '/signs/csv/'


def saveCSV(filename, entries):
	with open(save_path + filename, 'w') as file:
		writer = csv.writer(file, lineterminator='\n')
		writer.writerow(("Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"))
		writer.writerows(entries)

def concatCSVS():
	csv_format = path + '000{0:02d}/GT-000{0:02d}.csv'
	small_path = '000{0:02d}/'
	csv = []
	for i in range(43):
		p = csv_format.format(i)
		small_path_i = small_path.format(i)
		csv_i = load_csv(p)

		csv += ([[small_path_i + cell if j is 0 else cell for j,cell in enumerate(line)] for line in csv_i])

	return csv
_index = lambda num: random.randrange(0, num - 1)

def next_batch(num, index_func = _index):
	info = load_csv(save_path + 'signs.csv')
	num_samples = len(info)
	batch = []
	labels = []
	for _ in range(num):
		index = index_func(num_samples)
		Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId = info[index]
		print(Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId)
		img = Img.open(path + Filename).crop(*(int(c) for c in (Roi_X1, Roi_Y1, Roi_X2, Roi_Y2)))
		img.set_label(0)
		batch.append(img.arr2d)
		batch.append(img.one_hot)
	return batch, labels


def testHash():
	info = load_csv(save_path + 'signs.csv')
	n = hash_split(0.01, len(info))


	for i, q in enumerate(n):
		print(i, q)

	print()
	print('len:', len(info))
	print('hash:', len(n))
	print('set:', len(set(n)))
range_map = lambda input_start, input_end, output_start, output_end: lambda input: output_start + ((output_end - output_start) / (
input_end - input_start)) * (input - input_start)


def testBW():
	img = next_batch(1, lambda x: 500)[0]
	# img.show()
	img.convert('RGB')
	# i.show()
	# i =  img.convert('L')
	# w = list(np.asarray(i))
	# print(w)
	# q = list(i.getdata())
	# im = Image.fromarray(q)
	# im.show()
	# pix = numpy.array(i)
	# p = chain.from_iterable(pix)
	# print(list(p))
	# print(pix)
	# print(img.shape)
	l = img.arr2d[0:20, 0:20]
	#
	# print(list(l))
	# l = get_2d_list_slice(img.arr2d, 0, 10, 0, 10)


	img2 = Image.fromarray(l, 'RGB')
	img2.show()


	xcenter = img.shape[0] // 2
	ycenter = img.shape[1] // 2

	xmin = xcenter - 7
	xmax = xcenter + 7
	ymin = ycenter - 7
	ymax = ycenter + 7
	#
	# ymap = range_map(0, img.shape[0], 128, -127)
	xmap = range_map(0, img.shape[1], 128, -127)
	dmap = range_map(0, np.math.sqrt(img.shape[0]**2 + img.shape[1]**2), 128, -127)
	dimap = lambda x, y: dmap(np.math.sqrt(x ** 2 + y ** 2))
	#
	for y, y_arr in enumerate(img.arr2d):
		for x, _ in enumerate(y_arr):
			rgb = img.arr2d[y][x]
			offset = (dmap(np.math.sqrt(x ** 2 + y ** 2)), dmap(np.math.sqrt(x**2 + y**2)), 0)
			rgb_arr = []
			for i, n in enumerate(rgb):
				val = n + offset[i]
				if val < 0:
					val = 0
				elif val > 255:
					val = 255
				rgb_arr.append(val)
			img.arr2d[y][x] = rgb_arr

	#
	# 		if (abs(np.math.sqrt((x - xcenter)**2 + (y - ycenter)**2)- xmin) < 1):
	# 				img.arr2d[y][x] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
	#
	# 		if x > xmin and x < xmax and y > ymin and y < ymax:
	# 			rgb = img.arr2d[y][x]
	# 			offset = (100, -10, 10)
	# 			img.arr2d[y][x] = [n + offset[i] for i, n in enumerate(rgb)]
			if (x == xmin or x == xmax)and (ymin < y < ymax):
				img.arr2d[y][x] = [dimap(x, y), 0, 0]
			if (y == ymin or y == ymax) and (xmin < x < xmax):
				img.arr2d[y][x] = [dimap(x, y), 0, 0]
	#



	# vector =

	# arr2 = np.asarray(img.arr1d).reshape(img.shape)
	img.update()
	img.image.show()
				#

	# img2 = Image.fromarray(img.arr2d, 'RGB')
	# img2.show()
	# print(vector)

def testStuff():
	length = 1000
	test_indexes = hash_split(0.1, length)

	train_i = filter(lambda a: a not in test_indexes, range(length))



# testHash()
# e = concatCSVS()
# saveCSV('signs.csv', e)
testBW()



