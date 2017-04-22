import numpy as np
from PIL import Image
import os

from image.Image import Img

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

def draw2d(array2d:np.ndarray, xmin:int, ymin:int, xmax:int, ymax:int, color: [int] = (255, 0, 0)):
	a = array2d.copy()

	for i in range(ymin, ymax + 1):
		a[i][xmin] = color
		a[i][xmax] = color
	for i in range(xmin, xmax + 1):
		a[ymin][i] = color
		a[ymax][i] = color
	return a


# img = Image.open(random_pic_path)
# arr = np.asarray(img)
# arr = draw2d(arr, 10, 10, 200, 200)
# Image.fromarray(arr).show()

# for y, y_arr in enumerate(array2d):
# 	for x, val in enumerate(y_arr):
# 		if (x == xmin or x == xmax) and (ymin <= y <= ymax):
# 			a[y][x] = color
# 		elif (y == ymin or y == ymax) and (xmin <= x <= xmax):
# 			a[y][x] = color
