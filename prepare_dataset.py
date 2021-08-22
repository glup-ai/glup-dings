import os
from os import walk
import glob
from re import A
import numpy as np
from numpy.core.defchararray import mod
from numpy.lib.function_base import average
from numpy.lib.type_check import imag

from flask import Flask, render_template, Response
from mtcnn import MTCNN
from tensorflow import keras

import cv2
from facial_recognition import process_image

import time
images = []
labels = []

for dir in os.listdir('unhandled'):
    if(dir.startswith('.')):
        continue
    images.append([])
    new_images = glob.glob(f"unhandled/{dir}/*.png")

    for img in new_images:
        images[-1].append(np.array(cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)))

    labels.append(dir)

images = np.array(images)
labels = np.array(labels)

detector = MTCNN()

j = 0

for person in images:
    i = 1
    for img in person:
        cv2.imshow('image', img)
        cv2.waitKey(0) 

        image_total, faces = process_image(img, detector, debug=False)

        if len(faces) <= 0:
            continue
        else:
            
            for face in faces:
                cv2.imshow('image', face)
                cv2.waitKey(0) 
                #face =r[0]

                try:
                    resized = cv2.resize(face, (94,125), interpolation = cv2.INTER_AREA)
                except Exception as e:
                    continue
                print(resized)

                cv2.imwrite(f'dataset/{labels[j]}/{int(i)}.png', cv2.cvtColor(resized.astype('float32'), cv2.COLOR_RGB2BGR))
                i+=1
    j+=1
