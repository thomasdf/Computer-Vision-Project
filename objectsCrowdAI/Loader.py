import os
import sys
import csv
import random
import PIL
from PIL import Image

base_dir = os.path.dirname(os.path.dirname( __file__ ))
print(base_dir)
csvpath = base_dir + '/datasets/object-detection-crowdai/labels.csv'
print(csvpath)
imagefolder = base_dir + '/datasets/object-detection-crowdai/'

def loadCSV():
    res = []
    with open(csvpath) as file:
        reader = csv.reader(file)
        for row in reader:
            tuple = []
            for index in row:
                tuple.append(index)
            res.append(tuple)
    return res

info = loadCSV()
num_samples = len(info)

def rowToTuple(row):
    xmin = row[0]
    ymin = row[1]
    xmax = row[2]
    ymax = row[3]
    filename = row[4]
    label = row[5]
    url = row[6]
    return(xmin, ymin, xmax, ymax, filename, label, url)

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

def next_batch(num):
    for _ in range(num):
        index = random.randrange(0, num_samples-1)
        xmin, ymin, xmax, ymax, filename, label, url = rowToTuple(info[index])
        image = loadImage(filename)
        print(xmin, ymin, xmax, ymax, filename, label, url)

def loadImage(name):
    img = Image.open(imagefolder + name)
    img.show()
    return img


next_batch(1)