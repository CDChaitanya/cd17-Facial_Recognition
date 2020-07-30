# Facial Recgnition

import cv2
import numpy as np

############################################ TAKING THE DATASET THROUGH WEBCAM #####################################################

# LOAD haar face CLASSIFIER
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# LOAD FUNCTION
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image    
    
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray , scaleFactor=1.3 , minNeighbors=5)
    
    if faces is ():
        return None
    
    # Crop all faces found 
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h , x:x+w]
    return cropped_face

# INITIALIZE WEBCAM
cam = cv2.VideoCapture(0)
count = 0

# COLLECTING 100 SAMPLES OF YOUR FACE FROM WEBCAM INPUT
while True:
    _,frame = cam.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame) , (200,200))
        face = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)
        
        # SAVE FILE IN SPECIFIC DIR WITH UNIQUE NAME
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        
        # PUT COUNT ON IMAGE & DISPLA LIVE COUNT
        cv2.putText(img=face, text=str(count), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
        cv2.imshow('FACE CROPPER', face)
    else:
        print('FACE NOT FOUND')
        pass
    if cv2.waitKey(1) == 13 or count == 100: # IF YOU HIT "ENTER KEY" OR COUNT REACH 100 IT WILL BREAK
        break

cam.release()
cv2.destroyAllWindows()
print('SAMPLE COLLECTON COMPLETE')
  
############################################ TRAINING THE MODEL #####################################################  

from os import listdir
from os.path import isfile , join

# Get the training data we previously made
data_path = './faces/user/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path , f))] 

# Create arrays for training data and labels
Training_Data = []
Labels = [] 

# Open training images in our datapath
# Create a numpy array for training data
for i,files in enumerate(only_files):
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels , dtype=np.int32)

# Initialize Facial Recgnition
model = cv2.face.LBPHFaceRecognizer_create()

# Let's Train our Model
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

############################################ RUN THE FACIAL RECGNITION #####################################################  

def face_detector(img , size=0.5):
    # Converting image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if faces is ():
        return img , []
    for (x,y,w,h) in faces:
        cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,255), thickness=2)
        roi = img[y:y+h , x:x+w]
        roi = cv2.resize(roi, dsize=(200,200))
    return img , roi

# Opening Webcam 
cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    image , face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # PASS FACE TO PREDICTION MODEL
        # "results" comprises of a tuple containing the label and the confidence value 
        results = model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1-(results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
        
        cv2.putText(img=image, text=display_string, org=(100,120), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,120,150), thickness=2)
        
        if confidence > 75:
            cv2.putText(img=image, text='Unlocked', org=(250,450), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,0), thickness=2)
            cv2.imshow('Face Recognition', image)
        else:
            cv2.putText(img=image, text='Locked', org=(250,450), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), thickness=2)
            cv2.imshow('Face Recognition', image)            
    
    except:
        cv2.putText(img=image, text='No Face Found', org=(220,120), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        cv2.putText(img=image, text='Locked', org=(250,450), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        cv2.imshow('Face Recgnition', image)
        pass
    
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()     

################################################################################################################################  
        