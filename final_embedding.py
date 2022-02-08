import dlib
import cv2
import time
from os import listdir
from os.path import isdir
import openface
import face_recognition
import numpy
from numpy import asarray
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
start_time = time.time()

predictor_model = "shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
#face_detector=MTCNN()
face_aligner = openface.AlignDlib(predictor_model)

# Take the image file name from the command line

def extract_face(img_dir):
    image = cv2.imread(img_dir) 
        
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image)
    
    # Loop through each face we found in the image
    
    alignedFace = face_aligner.align(160, image, detected_faces[0], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
    alignedFace=asarray(alignedFace)
    
    return alignedFace


def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces


def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
X, y = load_dataset('train/')
emb_X=list()
emb_y=list()
for i,face_pixels in enumerate(X):
    embedding=face_recognition.face_encodings(face_pixels,known_face_locations=[(0, 160, 160, 0)])
   
    emb_X.append(embedding)
    emb_y.append(y[i])
    
emb_X=asarray(emb_X)
emb_X=numpy.squeeze(emb_X)
emb_y=asarray(emb_y)
print(emb_X.shape)
















