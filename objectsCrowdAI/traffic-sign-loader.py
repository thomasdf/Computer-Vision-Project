import csv
import random

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

def next_batch(num):
	info = loadCSV(save_path + 'signs.csv')
	num_samples = len(info)
	images = []
	for _ in range(num):
		index = random.randrange(0, num_samples - 1)
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

# next_batch(10)
testHash()
# e = concatCSVS()
# saveCSV('signs.csv', e)




