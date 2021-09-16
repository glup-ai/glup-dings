
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from deepface.commons import functions

def represent(img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'opencv', align = True):

	"""
	This function represents facial images as vectors.
	Parameters:
		img_path: exact image path, numpy array or based64 encoded images could be passed.
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.
		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.
			model = DeepFace.build_model('VGG-Face')
		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""

	if model is None:
		model = build_model(model_name)

	#---------------------------------

	#decide input shape
	input_shape =  input_shape_x, input_shape_y= functions.find_input_shape(model)

	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, align = align)

	#represent
	embedding = model.predict(img)[0].tolist()

	return embedding