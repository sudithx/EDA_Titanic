
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model
saved = joblib.load("iris_model.joblib")
model = saved["model"]
target_names = saved["target_names"]

app = FastAPI(title="Iris Model Serving API")


# Validate input values
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


# Home endpoint
@app.get("/")
def home():
    return {"message": "Iris prediction API is running"}


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "predicted_class": target_names[prediction],
        "class_id": prediction,
        "confidence": round(float(probabilities[prediction]), 4)
    }
