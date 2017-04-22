import numpy as np
from PIL import Image
from PIL import ImageDraw
import numpy as np
from learning.SlidingWindowV2 import slidy_mac_slideface, classic
import os
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


def shade2d(classified_img, size: int, intensity: int = 1):
    treshold = .90
    object = [(0,0,0,0), (0,255,255,intensity), (255,0,255, intensity), (255,255,0,intensity), (0,255,0, intensity), (0,0,0,0)]
    # 1. nothing = (0,0,0,0)
    # 2. car = (0,255,255,intensity)
    # 3. ped = (255,0,255, intensity)
    # 4. sign = (255,255,0,intensity)
    # 5. truck = (0,255,0, intensity)
    rect = Image.new('RGBA', (size, size))

    im = Image.open(random_pic_path)
    rect = Image.new('RGBA', (size, size))
    pdraw = ImageDraw.Draw(rect)
    for i in classified_img:
        offset = (i[0],i[1])
        object_index = 0
        if (i[2][i[2].argmax()] >= treshold):
            object_index = i[2].argmax() + 1
        pdraw.rectangle([0, 0, size, size],
                        fill=object[object_index], outline=object[object_index])
        im.paste(rect,offset, mask=rect)
    return np.array(im)






img = Image.open(random_pic_path)
arr = np.asarray(img)

print(shade2d(slidy_mac_slideface(arr, 60, 60, classic), 60,50))

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

