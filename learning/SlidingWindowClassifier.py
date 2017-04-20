import numpy as np

from image.Image import Img


def slidingwindowclassify(image: Img, stride: int, height: int, width: int, classifier: function):
	"""Slides a classifier over an image. That is: runs classifier for each height*width frame, stride apart"""
	imgwidth = image.shape[1]
	imgheight = image.shape[0]
	result = []
	if height > imgheight and width > imgwidth:
		raise Exception("Image smaller than classifier")
	horizontal = np.floor((imgwidth - width) / stride)
	vertical = np.floor((imgheight - height) / stride)

	ymin = 0
	ymax = height
	for _ in range(vertical):
		xmin = 0
		xmax = width
		for __ in range(horizontal):
			result.append(classify(image.crop(xmin, ymin, xmax, ymax), classifier))
			xmin = xmin + stride
			xmax = xmax + stride
		ymin = ymin + stride
		ymax = ymax + stride
	return result

def classify(image: Img, classifier: function):
	"""runs the classifier on the image"""
	image.normalize()
	return classifier(image.arr1d)
