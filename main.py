from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from io import BytesIO

# Charger le modèle
model = load_model('model.h5')
class_names = ["apple", "banana", "orange"]

# Créer l'application FastAPI
app = FastAPI()

# Configurer le répertoire des templates
templates = Jinja2Templates(directory="templates")

# Configuration du dossier d'uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def prepare_image(file_bytes):
    img = image.load_img(BytesIO(file_bytes), target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_bytes = await file.read()

    try:
        # Préparer l'image et faire la prédiction
        img_array = prepare_image(file_bytes)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        predicted_class_name = class_names[predicted_class]

        return JSONResponse({
            'prediction': predicted_class_name,
            'confidence': confidence
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir les fichiers statiques (si nécessaire)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
