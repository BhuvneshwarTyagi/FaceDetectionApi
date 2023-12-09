import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle5 as pickle

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
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
