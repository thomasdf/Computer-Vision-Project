import os
import PIL
import time
from PIL import Image

base_dir = os.path.dirname(os.path.dirname(__file__))

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498371963069978.jpg'

basewidth = 300
img = Image.open(random_pic_path)
# img.show()

print('ninja')
t0 = time.time()
wpercent = ( basewidth /float(img.size[0]))
hsize = int((float(img.size[1] ) * float(wpercent)))
img = img.resize((basewidth ,hsize), PIL.Image.ANTIALIAS)
print('kake', (time.time() - t0))
# img.show()
