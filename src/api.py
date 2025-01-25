"""
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

model_path ="models/my_model.keras"
app = FastAPI()
model = tf.keras.models.load_model(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return {"prediction": "sain" if np.argmax(prediction) == 0 else "malade"}
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Any
import io
from PIL import Image
import numpy as np
import tensorflow as tf  # Remplacez par PyTorch ou un autre framework selon le besoin

app = FastAPI()

# Charger votre modèle pré-entraîné
MODEL_PATH = "models/my_model.keras"  # Remplacez par le chemin réel vers votre modèle
model = tf.keras.models.load_model(MODEL_PATH)

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Vérifier si le fichier est une image
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG or PNG images are supported.")
        
        # Lire l'image téléchargée
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Prétraitement de l'image (adapter au modèle)
        image = image.resize((128, 128))  # Exemple : redimensionnement
        image_array = np.array(image) / 255.0  # Normalisation
        image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch

        # Faire une prédiction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        # Exemple de mapping des classes (adapter à votre modèle)
        class_names = ['Bacterial_spot', 
          'Early_blight', 
          'Late_blight', 
          'Leaf_Mold', 
          'Mosaic_virus',
          'Septoria_leaf_spot',
          'Spider_mites Two-spotted_spider_mite',
          'Target_Spot',
          'Tomato_healthy',
          'Yellow_Leaf_Curl_Virus']  # Remplacez par vos classes
        predicted_label = class_names[predicted_class]

        # Retourner la réponse
        return PredictionResponse(prediction=predicted_label, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Lancer l'application (si le script est exécuté directement)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

