import os
import time

import numpy as np
from PIL import Image
from PIL import ImageDraw

from image.Image import Img
from learning.SlidingWindowV2 import slide
from learning.thomasnetv2 import ThomasNet

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

def draw2d(array2d: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, color: [int] = (255, 0, 0)):
    a = array2d.copy()

    for i in range(ymin, ymax + 1):
        a[i][xmin] = color
        a[i][xmax] = color
    for i in range(xmin, xmax + 1):
        a[ymin][i] = color
        a[ymax][i] = color
    return a


def shade2d(im:Image , classified_img, size: int, intensity: int = 1, treshold: float=.90, scaled_shader: bool = True):
    object = [(0, 255, 255, intensity), (0,0,255,intensity), (255,0,0, intensity), (0,255,0,intensity)]
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
            color = object[object_index]
            color[3] = (color[3] * scale) if scaled_shader else color[3]
            pdraw.rectangle([0, 0, size, size], fill=color, outline=object[object_index])
            im.paste(rect,offset, mask=rect)
    return np.array(im)


def shade2dv2(shape: (), classified_img, size: int, intensity: int, treshold:float =.90):
    intens = intensity / 255.0
    object = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

    colors = [np.multiply(m, intens) for m in object]

    # 1. nothing = (0,0,0,0)
    # 2. car = (0,255,255,intensity)
    # 3. ped = (255,0,255, intensity)
    # 4. sign = (255,255,0,intensity)
    # 5. truck = (0,255,0, intensity)
    # rect = Image.new('RGBA', (size, size))
    arr2 = np.zeros(shape=(shape))

    for x, y, cls in classified_img:
        mx = cls.argmax()
        if mx > treshold:
            lable = int(cls[mx])
            l = colors[lable]

            arr2[y:(y + size), x:(x + size)] += l


    return arr2


def classicer(array: np.ndarray):
    darker = lambda t: 255 - t
    vfunc = np.vectorize(darker)
    midx = len(array) // 2
    a = vfunc(array[midx:midx + 4])

    return Img.static_normalized(a)


def out(img: Image, classifier, epoch: int, acc: float, treshold: float = .90):
    ttot = time.time()
    #
    # clsfy = np.vectorize(classifier)

    arr = np.asarray(img.convert('L'))

    tslide = time.time()
    coor, slices  = slide(arr, classifier.size, classifier.size)
    tslide = time.time() - tslide
    tclasfy = time.time()
    # c = []
    # for x, y, e in b:
    #     c.append((x, y, classifier(e)))
    r = classifier.run_nn(slices, epoch, acc)


    tclasfy = time.time() - tclasfy

    c = zip(coor, r)

    tshade = time.time()
    a = shade2d(img, c, classifier.size, 120, treshold)
    tshade = time.time() - tshade

    ttot = time.time() - ttot

    print('slide', tslide)
    print('classify', tclasfy)
    print('shade', tshade)
    print('tot', ttot)
    return a



if __name__ == '__main__':
    a = out(Image.open(random_pic_path), ThomasNet(), 3, 0.552)
    Image.fromarray(a).show()



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

