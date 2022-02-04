import dlib
import cv2
import time
import glob
import os
import openface
import face_recognition
start_time = time.time()



predictor_model = "shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)
"""
win = dlib.image_window()
win_aligned = dlib.image_window()
"""


# Take the image file name from the command line
#file_name = "index.jpg"
img_dir = "training_images"
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for i,f1 in enumerate(files): 
    image = cv2.imread(f1) 
        
    # Load the image
    
    #image=cv2.imread(file_name)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)
    
    #print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
    """
    # Show the desktop window with the image
    win.set_image(image)
    win_aligned.set_image(image)
    """
    # Loop through each face we found in the image
    for face_rect in detected_faces:
    
        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
       # print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
    
        # Draw a box around each face we found
        
        # Get the the face's pose
        #pose_landmarks = face_pose_predictor(image, face_rect)
        
        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(1000, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
        #aligned_landmarks=face_pose_predictor(alignedFace, face_rect)
        # Draw the face landmarks on the screen.
        
        alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
        emb=face_recognition.face_encodings((alignedFace))
        print(emb)
        cv2.imwrite("aligned_training_images/aligned_face_{}.jpg".format(i), alignedFace)
dlib.hit_enter_to_continue()
print("--- %s seconds ---" % (time.time() - start_time))