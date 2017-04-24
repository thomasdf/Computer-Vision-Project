# test_model.py

import numpy as np
# from grabscreen import grab_screen
# import cv2
# import time
# from directkeys import PressKey,ReleaseKey, W, A, S, D
# from alexnet import alexnet
# from getkeys import key_check

import random

import time

from main.alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

model = alexnet(WIDTH, HEIGHT, LR, 4)
model.load(MODEL_NAME)



def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:
            print('loop took {} seconds'.format(time.time()-last_time))


            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()

main()       










