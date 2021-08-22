import sys, time
import numpy as np
from flask import Flask, render_template, Response
from mtcnn import MTCNN
import cv2
from facial_recognition import process_image

class FacePosition:
    def __init__(self, x: int, y: int, last_seen=None, size=None):
        """
        x:
            x coordinate of the top left corner of the face.
        y:
            y coordinate of the top left corner of the face.

        last_seen:
            The time (in seconds) this face was last seen.

        size: [0, 1]
            The size of the face in fractions of the total image area.
            This corresponds to how far away the face is from the
            camera.
        """
        self.x = x
        self.y = y
        self.r = np.sqrt(x**2 + y**2)
        self.size = size
        
        if last_seen is None:
            """
            Experiencing trouble if i put time.time() as a default value
            for last_seen in the method signature.
            """
            self.last_seen = time.time()
        else:
            self.last_seen = last_seen

def tracking():
    """
    POSITION_TOL:
        How far a face can move in 1 frame and still be identified as
        the same face, [px]. Note that POSITION_TOL is scaled with the
        size of the face, the size corresponding to the distance from
        the camera to the face. A face closer to the camera may move a
        greater distance than a face far away.
    
    TIME_TOL:
        How long a face can be gone from the image before coming back
        and being identified as the same face, [s].
    """
    POSITION_TOL = 1000
    TIME_TOL = 4
    DEBUG = True

    saved_face_positions = []
    detector = MTCNN()
    colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0)}

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        try:
            rval, frame = vc.read()
            
            image_total, faces = process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detector, debug=DEBUG)
            total_amount_of_pixels = image_total.shape[0]*image_total.shape[1]

            for face, loc in faces:
                try:
                    """
                    Why this try-except?
                    """
                    resized = cv2.resize(face, (94, 125), interpolation = cv2.INTER_AREA)
                except:
                    continue
                
                x, y, width, height = loc   # x, y is the coordinate of the top left corner.
                new_face_position = FacePosition(
                    x = x,
                    y = y,
                    size = width*height/total_amount_of_pixels
                )

                is_it_a_new_face = True
                face_index = 0  # Needed for the initial face.

                for face_index in range(len(saved_face_positions)):
                    """
                    Compare the new face position to all saved face
                    positions.
                    """
                    face_position_diff = FacePosition(
                        x = np.abs(new_face_position.x - saved_face_positions[face_index].x),
                        y = np.abs(new_face_position.y - saved_face_positions[face_index].y),
                        last_seen = np.abs(new_face_position.last_seen - saved_face_positions[face_index].last_seen)
                    )

                    if (face_position_diff.r < new_face_position.size*POSITION_TOL) and (face_position_diff.last_seen < TIME_TOL):
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
                    text = f"hore {face_index}, size {saved_face_positions[face_index].size:.3f}",
                    org = (x, y),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
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