import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import cv2
import pickle
import time
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def update_github_file(repo_owner, repo_name, file_path, content, commit_message, access_token):
    print("update_github_file......................fgasd.gsdg.sd.ga.sdg.asd.g.asdg)
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    params = {
        "message": commit_message,
        "content": content
    }

    response = requests.put(url, headers=headers, json=params)

    if response.status_code == 200:
        return response.json()
    else:
        return None
        
@app.post("/train model")
async def predict():
    
    repo_owner = "BhuvneshwarTyagi"
    repo_name = "FaceDetectionApi"
    access_token = "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"  

    # Detect face for traing images
    # directoryPath="Train"
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # for folder in os.listdir(directoryPath):
    #     for image in os.listdir(folder):
    #         with open(os.path.join(folder, image)) as f:
    #             image=f.read()
    #             gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    #             for (x, y, w, h) in faces:
    #                 # Draw a rectangle around the face
    #                 cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #                 # Save the face region as an image
    #                 face_roi = image[y:y+h, x:x+w]
    #                 # New path where we want to save
    #                 file_path = os.path.join(f'{directoryPath}/{folder}', f'{os.path.join(folder, image)}')
    #                 print("File path is :",file_path)
    #                 cv2.imwrite(file_path, face_roi)
    #                 f.close()

    # # Detect face for Testing Images
    # directoryPath="Test"
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # for folder in os.listdir(directoryPath):
    #     for image in os.listdir(folder):
    #         with open(os.path.join(folder, image)) as f:
    #             image=f.read()
    #             gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    #             for (x, y, w, h) in faces:
    #                 # Draw a rectangle around the face
    #                 cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #                 # Save the face region as an image
    #                 face_roi = image[y:y+h, x:x+w]
    #                 # New path where we want to save
    #                 file_path = os.path.join(f'{directoryPath}/{folder}', f'{os.path.join(folder, image)}')
    #                 print("File path is :",file_path)
    #                 cv2.imwrite(file_path, face_roi)
    #                 f.close()


    # # # ----------------------- --------- start to train model -------------------------------------------------  

    TrainingImagePath='Train'
    TestingImagePath='Test'

    train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator()
    
    # Generating the Training Data
    training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical') 
    
    # Generating the Testing Data
    test_set = test_datagen.flow_from_directory(
        TestingImagePath,                               # Do change Here----------
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


    
    #------------------------------ Second Step to map face with face Id

    # class_indices have the numeric tag for each face
    TrainClasses=training_set.class_indices
 
    # Storing the face and the numeric tag for future reference
    ResultMap={}
    for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
        ResultMap[faceValue]=faceName
 
    # Saving the face map for future reference  or lebel of the images

    with open("lable.txt", 'wb') as fileWriteStream:
        pickle.dump(ResultMap, fileWriteStream)
 
    # The model will give answer as a numeric tag
    # This mapping will help to get the corresponding face name for it
    print("Mapping of Face and its ID",ResultMap)
 
    # The number of neurons for the output layer is equal to the number of faces
    OutputNeurons=len(ResultMap)
    print('\n The Number of output neurons: ', OutputNeurons)
    #-------------------------- third step train the model ------------------------------------------

 
 
    '''Initializing the Convolutional Neural Network'''
    classifier= Sequential()
 
    ''' STEP--1 Convolution
    # Adding the first layer of CNN
    # we are using the format (64,64,3) because we are using TensorFlow backend
    # It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
    '''
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
 
    '''# STEP--2 MAX Pooling'''
    classifier.add(MaxPool2D(pool_size=(2,2)))
 
    '''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
    classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
 
    classifier.add(MaxPool2D(pool_size=(2,2)))
 
    '''# STEP--3 FLattening'''
    classifier.add(Flatten())
 
    '''# STEP--4 Fully Connected Neural Network'''
    classifier.add(Dense(64, activation='relu'))
 
    classifier.add(Dense(OutputNeurons, activation='softmax'))
 
    '''# Compiling the CNN'''
    #classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
 
    ###########################################################

    # Starting the model training
    classifier.fit_generator(
                    training_set,
                    steps_per_epoch=5,
                    epochs=15,
                    validation_data=test_set,
                    validation_steps=10)

    
    classifier.save('newmodel.h5')

    # Update lable.txt on GitHub
    with open("lable.txt", 'rb') as file:
        content = file.read()
        update_github_file(repo_owner, repo_name, "lable.txt", content, "Update lable.txt", access_token)

    # Update newmodel.h5 on GitHub
    with open("newmodel.h5", 'rb') as file:
        content = file.read()
        update_github_file(repo_owner, repo_name, "newmodel.h5", content, "Update newmodel.h5", access_token)

    scores = classifier.evaluate_generator(test_set, steps=10)


    return JSONResponse(content={"Acuraccy": f"{scores}"}) 














