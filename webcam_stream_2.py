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

# Load dataset


for dir in os.listdir('dataset'):
    if(dir.startswith('.')):
        continue
    images.append([])
    new_images = glob.glob(f"dataset/{dir}/*.png")

    for img in new_images:
        images[-1].append(cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB))

    labels.append(dir)

images = np.array(images)
labels = np.array(labels)

persons= []

app = Flask(__name__) 
detector = MTCNN()

model = keras.models.load_model('splitted_twin_3')


for i in range(len(images)):
    features = model.predict(np.array(images[i]))
    persons.append(features)

persons = np.array(persons)


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

i = 20
while rval:
    i += 1
    rval, frame = vc.read()
    
    image_total, faces = process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detector, debug=False)

    for (face,loc) in faces:
        try:
            resized = cv2.resize(face, (94,125), interpolation = cv2.INTER_AREA)
        except Exception as e:
            continue

        #cv2.imwrite(f'dataset/{int(time.time())}.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        (x, y, width, height) = loc
        

        features = np.array(model.predict(np.array([resized]))[0])

        averages = []

        for person in persons:
            
            distances = np.linalg.norm(features-person, axis=1)
            print(distances.shape)
            print(distances)
                
            averages.append(np.average(distances))

        print(averages)
        index = np.argmin(averages)

        label = labels[index]

        cv2.putText(image_total,f"{label}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)


    cv2.imshow("preview", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyAllWindows()
vc.release()
