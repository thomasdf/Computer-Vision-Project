import numpy as np
import time
from PIL import Image
from PIL import ImageDraw

from image.Image import Img

from load import base_dir


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
			a = Img.static_normalized(arr.ravel())
			coordinates.append((x, y))
			slices.append(a)
	stacked_slice = np.vstack(slices)
	return coordinates, stacked_slice

def classify(array:np.ndarray, classifier:callable):
	return classifier(array)


def draw2d(array2d: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, color: [int] = (255, 0, 0)):
	a = array2d.copy()

	for i in range(ymin, ymax + 1):
		a[i][xmin] = color
		a[i][xmax] = color
	for i in range(xmin, xmax + 1):
		a[ymin][i] = color
		a[ymax][i] = color
	return a


def shade2d(im: Image, classified_img, size: int, intensity: int = 1, treshold: float = .90, scaled_shader: bool = True):
	object = [(0, 255, 255, intensity), (0, 0, 255, intensity), (255, 0, 0, intensity), (0, 255, 0, intensity)]
	dont_use_treshhold = treshold == -1
	# sign:turkis  ped:blue car:rød truck:grønn
	# 1. nothing = (0,0,0,0)
	# 2. car = (0,255,255,intensity)
	# 3. ped = (255,0,255, intensity)
	# 4. sign = (255,255,0,intensity)
	# 5. truck = (0,255,0, intensity)
	rect = Image.new('RGBA', (size, size))
	pdraw = ImageDraw.Draw(rect)
	for xy, cl in classified_img:
		x, y = xy
		offset = (x, y)
		object_index = cl.argmax()
		scale = cl[object_index]
		if (dont_use_treshhold or scale >= treshold):
			# object_index = cl.argmax()

			color = list(object[object_index])
			color[3] = int(color[3] * scale) if scaled_shader else color[3]
			pdraw.rectangle([0, 0, size, size], fill=tuple(color), outline=object[object_index])
			im.paste(rect, offset, mask=rect)
	return np.array(im)

if __name__ == '__main__':
	random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

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
