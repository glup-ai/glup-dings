import sys
import numpy as np
from flask import Flask, render_template, Response
from mtcnn import MTCNN
import cv2
from facial_recognition import process_image

def tracking():
    # images = []
    MAX_FACES = 20
    POSITION_TOL = 100
    labels = list(reversed(range(MAX_FACES)))
    face_positions = np.full(shape=MAX_FACES, fill_value=1e6, dtype=int)
    # app = Flask(__name__) 
    detector = MTCNN()

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

        for face, loc in faces:
            try:
                resized = cv2.resize(face, (94, 125), interpolation = cv2.INTER_AREA)
            except:
                continue
            #cv2.imwrite(f'dataset/{i}.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            (x, y, width, height) = loc
            face_position = np.linalg.norm([x, y])
            face_positions_diff = np.abs(face_position - face_positions)
            best_match_idx = np.argmin(face_positions_diff)
            best_match = face_positions_diff[best_match_idx]
            
            if best_match > POSITION_TOL:
                """
                New face.
                """
                face_label = labels[best_match_idx]

            cv2.putText(
                img = image_total,
                text = f"hore",
                org = (x, y),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 2,
                color = 255
            )

        cv2.imshow("preview", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
    vc.release()

tracking()