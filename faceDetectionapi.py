from fastapi import FastAPI,File, UploadFile
import cv2
import numpy as np
from starlette.responses import StreamingResponse
import io

app = FastAPI()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
@app.post("/detect_faces")
async def detect_faces(file: UploadFile = File(...)):
    image=await file.read()
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
            # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Save the face region as an image
        face_roi = image[y:y+h, x:x+w]
    _, img_encoded = cv2.imencode('.jpg', face_roi)
    img_bytes = img_encoded.tobytes()

    # Return the processed image
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
