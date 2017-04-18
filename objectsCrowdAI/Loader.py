import csv
import os
import random

import numpy as np

from objectsCrowdAI.Image import Img
from tools.SplitSet import hashSplit

base_dir = os.path.dirname(os.path.dirname( __file__ ))
print(base_dir)
csvpath = base_dir + '/datasets/object-detection-crowdai/labels.csv'
print(csvpath)
imagefolder = base_dir + '/datasets/object-detection-crowdai/'
dialect = None

def loadCSV(csvpath: str):
	res = []
	with open(csvpath) as file:
		dialect = csv.Sniffer().sniff(file.read(), delimiters=';,')
		file.seek(0)
		reader = csv.reader(file, dialect=dialect)
		for row in reader:
			tuple = []
			for index in row:
				tuple.append(index)
			res.append(tuple)
	return res

def defineSets(test_part: float, splitsetfunc = hashSplit):
	testindexes = splitsetfunc(test_part, num_samples)
	trainindexes = list(filter(lambda x: x not in testindexes, range(num_samples)))
	return (testindexes, trainindexes)

def labelToInt(label):
	if(label == "Pedestrian"):
		return 0
	if(label == "Car"):
		return 1
	if(label == "Truck"):
		return 2

def intToLabel(int):
	if(int == 0):
		return "Pedestrian"
	if(int == 1):
		return "Car"
	if(int == 2):
		return "Truck"

def next_batch_test(num):
	batch = next_batch(num, test_indexes) #test_indexes: array of all test-indexes
	return batch

def next_batch_train(num):
	batch = next_batch(num, train_indexes) #train_indexes: array of all train-indexes
	return batch


range_map = lambda input_start, input_end, output_start, output_end: lambda input: output_start + ((
                                                                                                   output_end - output_start) / (
	                                                                                                   input_end - input_start)) * (
                                                                                                  input - input_start)


def next_batch(num, set_indexes):
	batch = []
	for _ in range(num):
		# index = set_indexes[random.randrange(0, len(set_indexes))] #get random index in set
		index = 200
		xmin, ymin, xmax, ymax, filename, label, url = info[index]
		print(xmin, ymin, xmax, ymax, filename, label, url)
		image = Img.open(imagefolder + filename) #load image
		image.crop(int(xmin), int(ymin), int(xmax), int(ymax)) #crop object
		image.convert('L') #Convert to grayscale
		# image.show()


		imagearray = image.arr2d #convert to 1Darray

		arr1d = image.arr1d
		print(list(arr1d))
		normalize_map = range_map(0, 255, 0, 1)
		unnormalize_map = lambda x: int(round(range_map(0, 1, 0, 255)(x)))


		arr = image.normalized()
		arr_back = Img.denormalized(arr)


		Img.from_array1d(arr_back, image.shape, 'L').show()
		Img.from_array1d(arr1d   , image.shape, 'L').show()

		# print(len(arrNorm))

		batch.append(imagearray)
	return batch

info = loadCSV(csvpath)
# num_samples = len(info)
# test_indexes, train_indexes = defineSets(0.1)

n = next_batch(1, lambda x: 200)

# for m in n:
# 	Img.from_array2d(m, 'L').show()
