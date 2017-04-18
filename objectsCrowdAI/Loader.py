import os
import sys
import csv
import random
import PIL
from PIL import Image
from objectsCrowdAI.SplitSet import hashSplit

base_dir = os.path.dirname(os.path.dirname( __file__ ))
print(base_dir)
csvpath = base_dir + '/datasets/object-detection-crowdai/labels.csv'
print(csvpath)
imagefolder = base_dir + '/datasets/object-detection-crowdai/'
dialect = None

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

def defineSets(test_part: float, splitsetfunc = hashSplit()):
    testindexes = splitsetfunc(test_part, num_samples)
    trainindexes = range(0, num_samples).filter(lambda x: x not in testindexes)
    return (testindexes, trainindexes)



def rowToTuple(row):
    xmin = row[0]
    ymin = row[1]
    xmax = row[2]
    ymax = row[3]
    filename = row[4]
    label = row[5]
    url = row[6]
    # xmin, xmax, ymin, ymax, Frame, Label, Preview URL
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

def next_batch_test(num):
    return next_batch(num, test_indexes)

def next_batch_train(num):
    return next_batch(num, train_indexes)

def next_batch(num, set):
    batch = []
    for _ in range(num):
        index = set[random.randrange(0, len(set))] #get random index in set
        xmin, ymin, xmax, ymax, filename, label, url = info[index]
        print(xmin, ymin, xmax, ymax, filename, label, url)
        image = loadImage(filename)
        image = cropImage(image, xmin, ymin, xmax, ymax)
        batch.append(image)
    return batch


def cropImage(image, xmin, ymin, xmax, ymax):
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    left = xmin
    top = ymin
    bottom = ymax
    right = xmax
    crop_rectangle = (left, top, right, bottom) #left, top, right, bottom
    image = image.crop(crop_rectangle)
    print(image.size)
    return image

def loadImage(name):
    img = Image.open(imagefolder + name)
    return img

info = loadCSV(csvpath)
num_samples = len(info)
test_indexes, train_indexes = defineSets(0.1)