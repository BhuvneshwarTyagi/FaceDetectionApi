import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import cv2
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your face recognition model
model = tf.keras.models.load_model('face_recognition_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
@app.post("/predict")
async def predict(file: UploadFile = File('...')):
    contents = await file.read()
    
    # Convert raw bytes to image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
   # contents = await file.read()
        # Draw rectangles around the faces and save the images
    for (x, y, w, h) in faces:
            # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Save the face region as an image
        face_roi = image[y:y+h, x:x+w]
            #file_path = os.path.join(dataset, f'{count}face1.jpg')
           # print("File path is :",file_path)
        #cv2.imwrite(contents, face_roi)
    #contents=await face_roi.read()
    #image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((64, 64))  # Resize as needed
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Perform prediction using your model
    predictions = model.predict(image_array)
    map = pickle.load(open("model.txt",'rb'))
    # You may need to post-process the predictions based on your model
    # For example, convert the predictions to class labels or confidence scores

    # Return the prediction as JSON
    return JSONResponse(content={"predictions": map[np.argmax(predictions)]})
