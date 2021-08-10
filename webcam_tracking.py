import sys
import numpy as np
from flask import Flask, render_template, Response
from mtcnn import MTCNN
import cv2
from numpy.lib.npyio import save
from facial_recognition import process_image

class FacePosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = np.sqrt(x**2 + y**2)

def tracking():
    POSITION_TOL = 100  # How far a face can move in 1 frame and still be identified as the same face.
    DEBUG = True

    saved_face_positions = []
    # app = Flask(__name__) 
    detector = MTCNN()
    colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0)}

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # i = 20
    while rval:
        try:
            # i += 1
            rval, frame = vc.read()
            
            image_total, faces = process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detector, debug=DEBUG)

            for face, loc in faces:
                try:
                    """
                    Why this try-except?
                    """
                    resized = cv2.resize(face, (94, 125), interpolation = cv2.INTER_AREA)
                except:
                    continue
                #cv2.imwrite(f'dataset/{i}.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                
                x, y, width, height = loc   # x, y is the coordinate of the top left corner.
                new_face_position = FacePosition(x, y)

                is_it_a_new_face = True
                face_index = 0  # Needed for the initial face.

                for face_index in range(len(saved_face_positions)):
                    """
                    Compare the new face position to all saved face
                    positions.
                    """
                    face_positions_diff = np.abs(new_face_position.r - saved_face_positions[face_index].r)

                    if face_positions_diff < POSITION_TOL:
                        """
                        Not a new face!
                        """
                        saved_face_positions[face_index] = new_face_position
                        is_it_a_new_face = False
                        break
                    
                if is_it_a_new_face:
                    """
                    Add new face.
                    """
                    saved_face_positions.append(new_face_position)

                cv2.putText(
                    img = image_total,
                    text = f"hore {face_index}",
                    org = (x, y),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 2,
                    color = 255
                )

            if DEBUG:
                for face in saved_face_positions:
                    """
                    Add arrows to indicate all saved face positions.
                    """
                    cv2.arrowedLine(
                        img = image_total,
                        pt1 = (0, 0),
                        pt2 = (face.x, face.y),
                        color = colors['red'],
                        thickness = 2,
                        line_type = 8,
                        shift = 0,
                        tipLength = 0.1
                    )

            cv2.imshow("preview", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            vc.release()
            print("Program ended.")
            sys.exit()

    cv2.destroyAllWindows()
    vc.release()

tracking()