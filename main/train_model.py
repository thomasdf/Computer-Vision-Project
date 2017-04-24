# train_model.py

import numpy as np
from PIL import Image

from load import base_dir
from main.Main import new_out
from main.alexnet import alexnet

random_pic_path = base_dir + '/datasets/object-detection-crowdai/1479498442970431250.jpg'


def train(size: int):
    WIDTH = size
    HEIGHT = size
    LR = 1e-3
    EPOCHS = 10

    img = Image.open(random_pic_path)

    MODEL_NAME = 'cvp-{}-{}-{}-epochs-data.model'.format(LR, 'ravnanet', EPOCHS)

    save_dir = base_dir + '/savedmodels/ravnanet'

    data_set_path = base_dir + '/load/balanced_data_set.npy'

    model = alexnet(WIDTH, HEIGHT, LR, 4)

    hm_data = 2
    for i in range(EPOCHS):
        for j in range(1, hm_data + 1):
            train_data = np.load(data_set_path)

            train = train_data[:-100]
            test = train_data[-100:]

            X = np.array([i[0].reshape(size, size) for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
            Y = [i[1] for i in train]


            # print(Y[0])
            # Image.fromarray(X[0]).show()

            test_x = np.array([i[0].reshape(size, size) for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
            test_y = [i[1] for i in test]

            model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
            model.save(MODEL_NAME)

            print('\n\n\nDONE EPOCH',i , 'hm load', j)
            a = new_out(img, size, model)
            Image.fromarray(a.shape(size, size)).show()


# C:\Users\kiwi\AppData\Local\Programs\Python\Python35\python.exe -m tensorflow.tensorboard --logdir=foo:C:/Users/kiwi/IdeaProjects/Computer-Vision-Project/main/log

if __name__ == '__main__':
    train(32)



