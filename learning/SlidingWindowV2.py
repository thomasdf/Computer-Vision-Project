import numpy as np
import time
from PIL import Image
import os

from image.Image import Img

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

def slidy_mac_slideface(array:np.ndarray, stride: int, size: int, classifier: callable):

	classified_img = []
	for y in range(0, len(array) - size, stride):

		for x in range(0, len(array[y]) - size, stride):

			arr = array[y:(y + size), x:(x + size)]

			# a = draw2d(a, x, y, (x + size), (y + size))

			result_arr = classifier(arr.ravel())

			classified_img.append((x, y, result_arr))


	return classified_img

def classic(array:np.ndarray):
	return Img.static_normalized(array[0:4])


def slide(array: np.ndarray, stride: int, size: int):
	coordinates = []
	slices = []

	for y in range(0, len(array) - size, stride):

		for x in range(0, len(array[y]) - size, stride):
			arr = array[y:(y + size), x:(x + size)]

			# a = draw2d(a, x, y, (x + size), (y + size))

			coordinates.append((x, y))
			slices.append(arr.ravel())
	stacked_slice = np.vstack(slices)
	return coordinates, stacked_slice

def classify(array:np.ndarray, classifier:callable):
	return classifier(array)


if __name__ == '__main__':

	img = Image.open(random_pic_path)
	arr = np.asarray(img)
	# img.show()
	for i in range(0, 30, 3):
		n = 30 + i
		t0 = time.time()
		slidy_mac_slideface(arr, n, 224, classic)
		print(n, time.time() - t0)


# for m in n:
# 	print(m)

# Image.fromarray(arr).show()
