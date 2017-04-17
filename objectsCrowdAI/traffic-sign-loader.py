import csv
import random

from PIL import Image

from objectsCrowdAI.Loader import base_dir, loadCSV, loadImage, cropImage

path = base_dir + '/datasets/traffic-signs/GTSRB/Final_Training/Images'
csvpath = base_dir + '/datasets/object-detection-crowdai/labels.csv'


def concatCSVS():
	csv_format = path + '/000{0:02d}/GT-000{0:02d}.csv'
	small_path = path + '/000{0:02d}/'
	csv = []
	for i in range(43):
		p = csv_format.format(i)
		small_path_i = small_path.format(i)
		csv_i = loadCSV(p)[1:]

		csv += ([[small_path_i + cell if j is 0 else int(cell) for j,cell in enumerate(line)] for line in csv_i])

	return csv


def next_batch(num):
	info = concatCSVS()
	num_samples = len(info)
	for _ in range(num):
		index = random.randrange(0, num_samples - 1)
		print(info[index])
		Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId = info[index]
		print(Filename, Width, Height, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2, ClassId)
		image = Image.open(Filename)
		image = cropImage(image, Roi_X1, Roi_Y1, Roi_X2, Roi_Y2)
		image.show()


next_batch(1)
