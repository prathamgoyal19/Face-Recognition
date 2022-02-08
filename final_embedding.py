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
import math

def main():
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    #face_detector=MTCNN()
    face_aligner = openface.AlignDlib(predictor_model)
    
    X, y = load_dataset('archive(1)/')
    X=numpy.squeeze((X))
    savez_compressed('embeddings_archive.npz',X,y)
    
def extract_face(img_dir, face_detector=dlib.get_frontal_face_detector(),face_aligner = openface.AlignDlib("shape_predictor_68_face_landmarks.dat") ):
    image = cv2.imread(img_dir) 
        
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image)
    if len(detected_faces)==0:
        return 0
    # Loop through each face we found in the image
    
    alignedFace = face_aligner.align(160, image, detected_faces[0], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
    alignedFace=asarray(alignedFace)
    embedding=face_recognition.face_encodings(alignedFace,known_face_locations=[(0, 160, 160, 0)])
    return embedding


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        if face==0:
            continue
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


if __name__ == "__main__":
    main()
