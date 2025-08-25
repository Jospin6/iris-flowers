from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from typing import List
import sys
import os

# Ajouter le chemin parent pour importer les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.ann import IrisANN
from src.training.trainer import IrisTrainer

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Charger le modèle au démarrage
@app.on_event("startup")
async def load_model():
    global model, device, class_names
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le modèle
    trainer, model = IrisTrainer.load_model('models/iris_model.pth', device)
    model.eval()
    
    # Noms des classes
    class_names = ['setosa', 'versicolor', 'virginica']
    
    print("Modèle chargé avec succès!")

# Modèle Pydantic pour la requête
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: List[float]
    class_names: List[str]

@app.get("/")
async def root():
    return {"message": "Iris Classification API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Convertir en tensor et ajouter une dimension batch
        input_data = np.array([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]).astype(np.float32)
        
        # Ajouter une dimension batch (shape: [1, 4] au lieu de [4])
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)  # Notez dim=1 maintenant
            confidence, predicted = torch.max(probabilities, 1)  # Notez dim=1
            
            return PredictionResponse(
                prediction=class_names[predicted.item()],
                confidence=confidence.item(),
                probabilities=probabilities.cpu().numpy()[0].tolist(),  # Prendre le premier élément du batch
                class_names=class_names
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        inputs = []
        for features in features_list:
            inputs.append([
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ])
        
        # Déjà en format batch, pas besoin de unsqueeze
        input_tensor = torch.FloatTensor(inputs).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            results = []
            for i in range(len(predictions)):
                results.append({
                    "prediction": class_names[predictions[i].item()],
                    "confidence": confidences[i].item(),
                    "probabilities": probabilities[i].cpu().numpy().tolist()
                })
            
            return {"predictions": results}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)