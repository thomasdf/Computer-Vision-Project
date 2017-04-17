import csv
import random
from itertools import chain

import numpy
import numpy as np
from PIL import Image

from objectsCrowdAI.Loader import base_dir, loadCSV
from objectsCrowdAI.SplitSet import hashSplit

path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images/'
save_path = base_dir + '/objectsCrowdAI/csv/'

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
		csv_i = loadCSV(p)[1:]

		csv += ([[small_path_i + cell if j is 0 else cell for j,cell in enumerate(line)] for line in csv_i])

	return csv
_index = lambda num: random.randrange(0, num - 1)

def next_batch(num, index_func = _index):
	info = loadCSV(save_path + 'signs.csv')
	num_samples = len(info)
	images = []
	for _ in range(num):
		index = index_func(num_samples)
		Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId = info[index]
		print(Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId)
		image = Image.open(path + Filename).crop((int(c) for c in (Roi_X1, Roi_Y1, Roi_X2, Roi_Y2)))
		images.append(image)
	return images

def testHash():
	info = loadCSV(save_path + 'signs.csv')
	n = hashSplit(0.01, len(info))


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
	i =  img.convert('RGB')
	w = list(np.asarray(i))
	# print(w)
	q = list(i.getdata())
	# im = Image.fromarray(q)
	# im.show()
	# pix = numpy.array(i)
	# p = chain.from_iterable(pix)
	# print(list(p))
	# print(pix)

	arr2d = np.array(i)


	shape = arr2d.shape

	xmin = shape[0] // 3
	xmax = shape[0] // 3 * 2
	ymin = shape[1] // 3
	ymax = shape[1] // 3 * 2

	xcenter = shape[0] // 2
	ycenter = shape[1] // 2
	xmap = range_map(0, shape[0], 0, 40)
	ymap = range_map(0, shape[1], 128, -127)

	for y, y_arr in enumerate(arr2d):
		for x, _ in enumerate(y_arr):
			rgb = arr2d[y][x]
			offset = (ymap(x), 0, 0)
			rgb_arr = []
			for i, n in enumerate(rgb):
				val = n + offset[i]
				if val < 0:
					val = 0
				elif val > 255:
					val = 255
				rgb_arr.append(val)
			arr2d[y][x] = rgb_arr


		# if (abs(np.math.sqrt((x - xcenter)**2 + (y - ycenter)**2)- xmin) < 1):
			# 	arr2d[y][x] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
			#
			# if x > xmin and x < xmax and y > ymin and y < ymax:
			# 	rgb = arr2d[y][x]
			# 	offset = (100, -10, 10)
			# 	arr2d[y][x] = [n + offset[i] for i, n in enumerate(rgb)]
			# if (x == xmin or x == xmax)and (y > ymin and y < ymax):
			# 	arr2d[y][x] = [0, 0, 100]
			# if (y == ymin or y == ymax) and (x > xmin and x < xmax):
			# 	arr2d[y][x] = [0, 0, 100]



	arr1d = arr2d.ravel()
	vector = np.matrix(arr1d)

	# vector =

	arr2 = np.asarray(vector).reshape(shape)

	img2 = Image.fromarray(arr2, 'RGB')
	img2.show()
	# print(vector)



# testHash()
# e = concatCSVS()
# saveCSV('signs.csv', e)
testBW()



