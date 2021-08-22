import os
from os import walk
import glob
import numpy as np
from numpy.lib.type_check import imag

from flask import Flask, render_template, Response
from mtcnn import MTCNN
from tensorflow import keras

import cv2
from facial_recognition import process_image

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

print(labels)

app = Flask(__name__) 
detector = MTCNN()

model = keras.models.load_model('trained_model')

print(model.summary())
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
    
    image_total, faces = process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detector, debug=True)

    for (face,loc) in faces:
        try:
            resized = cv2.resize(face, (94,125), interpolation = cv2.INTER_AREA)
        except Exception as e:
            continue
        #cv2.imwrite(f'dataset/{i}.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        (x, y, width, height) = loc

        pred_avg = []

        for i in range(len(images)):

            array = np.array([resized]*len(images[i]))
            preds = model.predict([array,np.array(images[i])])

            pred_avg.append(np.average(preds))

        pred_avg = np.array(pred_avg)
        label = labels[pred_avg.argmax()]
        
        cv2.putText(image_total,f"{label} c: {np.amax(pred_avg):.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)


    cv2.imshow("preview", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyAllWindows()
vc.release()
