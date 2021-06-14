<<<<<<< HEAD
import numpy as np
import cv2
import time

def process_image(image, detector, ratio=0.752, min_confidence=0.9, debug=False):
    """
    Process a single image.
    Parameters
    ----------
    image : 3d numpy array
        Input image in BGR-space.
    output_directory : string
        Path to the directory where the processed images will be saved.
    detector : ?
        Passed as argument to avoid overhead.
    """
     
=======
import time
import sys
import os
import numpy as np
import cv2
from mtcnn import MTCNN

# TODO: Implement adjustable face crop.

MIN_CONFIDENCE = 0.97   # Skip all faces under this confidence.
RAW_DIRECTORY = "fr-imgs-input"
PROCESSED_DIRECTORY = "fr-imgs-output"
PRINT_INFO = True   # Turn debug info on / off.
SUPPORTED_IMAGE_FORMATS = ("jpg", "jpeg")
SUPPORTED_VIDEO_FORMATS = ("mov",)

def video_to_images(video_fname):
    """
    Save each frame of a video as individual jpg images.  Create a
    directory with the same name as the video file and save the frames
    there.

    Warning: Overwrites existing images if run several times for the
    same video.

    TODO: Input video path instead of only filename.
    """
    if (extension := video_fname.split(".")[-1]) not in SUPPORTED_VIDEO_FORMATS:
        msg = f"Extension '{extension}' is not yet implemented."
        raise NotImplementedError(msg)

    cap = cv2.VideoCapture(RAW_DIRECTORY + video_fname)
    
    if (video_fname.count(".") > 1):
        msg = f"Video filename cannot contain any periods '.' except for the file extension."
        raise NotImplementedError(msg)
    
    video_fname_no_extension = video_fname.split('.')[0]
    image_output_directory = f"{RAW_DIRECTORY}{video_fname_no_extension}"
    os.system(f"mkdir {image_output_directory}")    # TODO: check if the directory already exists (to suppress mkdir message).

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_fname = f"{video_fname_no_extension}_{frame_counter:08d}.jpg"
        
        if ret:
            cv2.imwrite(
                filename = f"{image_output_directory}/{frame_fname}",
                img = frame
            )
        else: break
        frame_counter += 1


def process_image(image, image_fname, output_directory, detector):
    """
    Process a single image.

    Parameters
    ----------
    image : ?
        Prob. 3d numpy array.

    image_fname : string
        Name of the image. Does not need to be the actual filename.

    output_directory : string
        Path to the directory where the processed images will be saved.

    detector : ?
        Passed as argument to avoid overhead.
    """
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0
    timing = time.time()
    results = detector.detect_faces(image)
    timing = time.time() - timing

    image_total = np.copy(image)    # Avoid blue frames when saving cropped faces.
    
<<<<<<< HEAD
    n_faces = len(results)

    faces = []
    
    if debug:
        print("========================")
=======
    fname_out = image_fname.split(".")[0] + "_faces"  # Construct output filename.
    n_faces = len(results)
    
    if PRINT_INFO:
        print("========================")
        print(f"{image_fname=}")
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0
        print(f"{image.shape=}")
        print(f"{timing=}")
        print(f"{n_faces=}")
        print(f"image size: {image.size*image.itemsize*1e-6:.1f} MB")

    if not results:
<<<<<<< HEAD
        if(debug):
            print(f"No faces detected in image. Skipping.")
=======
        print(f"No faces detected in image {image_fname}. Skipping.")
        return
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0

    for face_idx, result in enumerate(results):
        """
        Save crop of each face.
        """
<<<<<<< HEAD
        if result['confidence'] < min_confidence:
            if debug:
                print(f"Face {face_idx + 1} of {n_faces} skipped with confidence: {result['confidence']}")
            continue
        
        if debug:
            print(f"{result['confidence']}")
        
  
        keypoints = result['keypoints']
        (x, y, width, height) = result['box']
        width_delta = int(((height * ratio) - width)/2)

        image_cropped = image[
                        y:y + height,
                        x-width_delta:x + width + width_delta, :
                    ]

        faces.append([image_cropped,result['box']])            
    
        cv2.rectangle(
            img = image_total,
            pt1 = (x, y),
            pt2 = (x + width, y + height),
=======
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
                filename = f"{output_directory}/crop/{fname_out}_crop_{face_idx}.jpg",
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
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0
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
<<<<<<< HEAD

=======
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0
        cv2.circle(image_total, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image_total, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
<<<<<<< HEAD

    if debug:
        print("========================\n")
    
    return image_total, faces

=======
        
    cv2.imwrite(f"{output_directory}/full/{fname_out}.jpg", cv2.cvtColor(image_total, cv2.COLOR_RGB2BGR))
    
    if PRINT_INFO:
        print("========================\n")



def image_loop(input_path, read_every_n_images=10):
    """
    Loop over all images in 'input_path'.

    Parameters
    ----------
    input_path : string
        Path to the directory where the images are stored.

    read_every_n_images : int
        Read only every n image, skip all others.
    """
    if input_path[-1] == "/": input_path = input_path[:-1]  # Remove trailing '/'.
    
    detector = MTCNN()
    input_directory = input_path.split("/")[-1]
    output_path = f"{PROCESSED_DIRECTORY}/{input_directory}"
    output_path_full = f"{PROCESSED_DIRECTORY}/{input_directory}/full"
    output_path_crop = f"{PROCESSED_DIRECTORY}/{input_directory}/crop"

    if not os.path.isdir(output_path):
        """
        Create directory for processed images.
        """
        os.system(f"mkdir {output_path}")
    
    if not os.path.isdir(f"{output_path_full}"):
        os.system(f"mkdir {output_path_full}")

    if not os.path.isdir(output_path_crop):
        os.system(f"mkdir {output_path_crop}")

    image_counter = 0
    for image_fname in sorted(os.listdir(input_path)):
        """
        Loop over all images in 'input_path'.
        """
        if image_fname.split(".")[-1] not in SUPPORTED_IMAGE_FORMATS:
            """
            Read only supported formats.
            """
            continue
        
        if (image_counter%read_every_n_images != 0):
            """
            Read only every n'th image.
            """
            image_counter += 1
            continue

        image = cv2.cvtColor(cv2.imread(f"{input_path}/{image_fname}"), cv2.COLOR_BGR2RGB)
        process_image(image, image_fname, output_path, detector)
        image_counter += 1


if __name__ == "__main__":
    input_path = f"{RAW_DIRECTORY}/standalone_images"
    # input_path = f"{RAW_DIRECTORY}/IMG_6030_720p"
    image_loop(input_path, read_every_n_images=1)
    # video_to_images()
    pass
>>>>>>> d206a76c1d9a27a11222aa666861d0cff7cafaf0
