import time
import os
import numpy as np
import cv2
from mtcnn import MTCNN

# TODO: Implement adjustable face crop.

MIN_CONFIDENCE = 0.97   # Skip all faces under this confidence.
RAW_DIRECTORY = "fr-imgs-input/"
PROCESSED_DIRECTORY = "fr-imgs-output/"
PRINT_INFO = True   # Turn debug info on / off.

detector = MTCNN()

for fname_img in sorted(os.listdir(RAW_DIRECTORY)):
    """
    Loop over all images in 'RAW_DIRECTORY'.
    """
    timing = time.time()
    image = cv2.cvtColor(cv2.imread(RAW_DIRECTORY + fname_img), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    timing = time.time() - timing

    image_total = np.copy(image)
    
    fname_out = fname_img.split(".")[0] + "_faces"  # Construct output filename.
    n_faces = len(results)
    
    if PRINT_INFO:
        print("========================")
        print(f"{fname_img=}")
        print(f"{image.shape=}")
        print(f"{timing=}")
        print(f"{n_faces=}")
        print(f"image size: {image.size*image.itemsize*1e-6:.1f} MB")

    if not results:
        print(f"No faces detected in image {fname_img}. Skipping.")
        continue

    for face_idx, result in enumerate(results):
        """
        Save crop of each face.
        """
        if result['confidence'] < MIN_CONFIDENCE:
            if PRINT_INFO:
                print(f"Face {face_idx + 1} of {n_faces} skipped with confidence: {result['confidence']}")
            continue
        
        if PRINT_INFO:
            print(f"{result['confidence']}")
        
        bounding_box = result['box']
        keypoints = result['keypoints']

        try:
            cv2.imwrite(    # Save crop of each face.
                filename = f"{PROCESSED_DIRECTORY}/{fname_out}_crop_{face_idx}.jpg",
                img = cv2.cvtColor( # Change color space.
                    src = image[
                        bounding_box[1]:bounding_box[1] + bounding_box[3],
                        bounding_box[0]:bounding_box[0] + bounding_box[2], :
                    ],
                    code = cv2.COLOR_RGB2BGR
                )
            )
        except cv2.error:
            """
            Unsure why this error occurs. A hint is the col start index
            which is negative.
            """
            print("DEBUG--------")
            print(f"Unknown error saving crop of face {face_idx + 1} of {n_faces}.")
            print(f"row: {bounding_box[1]}:{bounding_box[1] + bounding_box[3]}")
            print(f"col: {bounding_box[0]}:{bounding_box[0] + bounding_box[2]}")
            print("DEBUG_END----")
    
        cv2.rectangle(
            img = image_total,
            pt1 = (bounding_box[0], bounding_box[1]),
            pt2 = (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            color = (0, 155, 255),
            thickness = 2
        )
        cv2.circle(
            img = image_total,
            center = (keypoints['left_eye']),
            radius = 2,
            color = (0, 155, 255),
            thickness = 2
        )
        cv2.circle(image_total, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
        
    cv2.imwrite(f"{PROCESSED_DIRECTORY}_{fname_out}.jpg", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))
    
    if PRINT_INFO:
        print("========================\n")
