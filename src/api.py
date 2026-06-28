import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "models/delivery_time_model.pkl"

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="API for predicting food delivery time using a trained ML model",
    version="1.0.0"
)


# -----------------------------
# Load the model at startup
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at: {MODEL_PATH}. "
        "Please train the model and save it before starting the API."
    )

model = joblib.load(MODEL_PATH)


# -----------------------------
# Request schema
# -----------------------------
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Distance in kilometers")
    items_count: int = Field(..., gt=0, description="Number of items in the order")
    is_peak_hour: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    traffic_level: int = Field(..., ge=1, le=3, description="1 = low, 2 = medium, 3 = high")


# -----------------------------
# Response schema
# -----------------------------
class PredictionResponse(BaseModel):
    delivery_time_min: float


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        input_df = pd.DataFrame([{
            "distance_km": request.distance_km,
            "items_count": request.items_count,
            "is_peak_hour": request.is_peak_hour,
            "traffic_level": request.traffic_level
        }])

        prediction = model.predict(input_df)[0]

        return PredictionResponse(
            delivery_time_min=round(float(prediction), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
