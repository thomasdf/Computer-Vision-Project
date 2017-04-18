import csv
import os
import random
from image.Image import Img
from tools.SplitSet import hashSplit

base_dir = os.path.dirname(os.path.dirname( __file__ ))
print(base_dir)
csvpath = base_dir + '/datasets/object-detection-crowdai/labels.csv'
print(csvpath)
imagefolder = base_dir + '/datasets/object-detection-crowdai/'
dialect = None

labels = ["Pedestrian", "Car", "Truck"]

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

def label_to_index(label):
	label.index(label) + 43

def index_to_label(int):
	return labels[int - 43]

def next_batch_test(num):
	batch = next_batch(num, test_indexes) #test_indexes: array of all test-indexes
	return batch

def next_batch_train(num):
	batch = next_batch(num, train_indexes) #train_indexes: array of all train-indexes
	return batch


def next_batch(num, set_indexes):
	batch = []
	for _ in range(num):
		index = set_indexes[random.randrange(0, len(set_indexes))] #get random index in set
		xmin, ymin, xmax, ymax, filename, label, url = info[index]
		print(xmin, ymin, xmax, ymax, filename, label, url)
		image = Img.open(imagefolder + filename) #load image
		image.crop(int(xmin), int(ymin), int(xmax), int(ymax)) #crop object
		image.convert('L') #Convert to grayscale
		image.label[label_to_index(label)] = 1
		# image.show()


		imagearray = image.arr2d #convert to 1Darray

		arr1d = image.arr1d


		arr = image.normalized()
		arr_back = Img.denormalized(arr)

		# arrCroped = image.croped_arr(0, 100, 0, 100)
		# print(arrCroped)
		# print(type(arrCroped))

		# print(image.shape)
		image.show()
		for _ in range(1):
			allah = image.rand_crop(1200, 1000)
			Img.from_array2d(allah, 'L').show()

		# print(len(arrNorm))

		batch.append(imagearray)
	return batch

info = loadCSV(csvpath)
num_samples = len(info)
test_indexes, train_indexes = defineSets(0.1)

n = next_batch(1, lambda x: 200)

# for m in n:
# 	Img.from_array2d(m, 'L').show()
