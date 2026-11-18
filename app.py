
from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI application
app = FastAPI(title="ML Lab - Iris Prediction API")

# Load trained model
model = joblib.load("model.pkl")

# Iris flower categories
flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to ML Lab API"
    }

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: list[float]):

    if len(data) != 4:
        return {
            "error": "Please provide exactly 4 features"
        }

    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)

    flower = flower_names[int(prediction[0])]

    return {
        "prediction": flower
    }
