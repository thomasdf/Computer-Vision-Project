import numpy as np
import time
from PIL import Image
import os

from image.Image import Img
from tools.ArrayTool import draw2d

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

def slidy_mac_slideface(array:np.ndarray, stride: int, size: int, classifier: callable):

	classified_img = []
	# a = array.copy()
	for y in range(0, len(array) - size, stride):

		for x in range(0, len(array[y]) - size, stride):

			arr = array[y:(y + size), x:(x + size)]

			# a = draw2d(a, x, y, (x + size), (y + size))

			result_arr =  classifier(arr.ravel())



			classified_img.append((x, y, result_arr))

	# Image.fromarray(a).show()

	return classified_img

def classic(array:np.ndarray):
	return Img.static_normalized(array[0:4])



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
