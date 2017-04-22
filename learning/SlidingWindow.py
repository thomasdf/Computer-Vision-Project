import numpy as np

from image.Image import Img
from image import Image


def slidingwindowclassify(image: Img, stride: int, height: int, width: int, classifier: callable):
	"""Slides a classifier over an image. That is: runs classifier for each height*width frame, stride apart"""
	image.normalize()
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
			result.append(classify(Img.from_image(image.image).crop(xmin, ymin, xmax, ymax), classifier)) #not efficient at all...
			xmin = xmin + stride
			xmax = xmax + stride
		ymin = ymin + stride
		ymax = ymax + stride
	return result

def classify(image: Img, classifier: callable):
	"""runs the classifier on the image"""
	return classifier(image.arr1d)
