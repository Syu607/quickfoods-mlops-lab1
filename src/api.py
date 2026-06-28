import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "models/delivery_time_model.pkl"
LOG_DIR    = "logs"
LOG_PATH   = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="API with prediction logging for monitoring",
    version="2.0.0"
)

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
os.makedirs(LOG_DIR, exist_ok=True)


class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)


class PredictionResponse(BaseModel):
    delivery_time_min: float


def log_prediction(request_data: dict, prediction: float):
    """Append one JSON line per prediction to the log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_data,
        "prediction": prediction,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check():
    log_count = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            log_count = sum(1 for _ in f)
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "predictions_logged": log_count,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        input_dict = {
            "distance_km": request.distance_km,
            "items_count": request.items_count,
            "is_peak_hour": request.is_peak_hour,
            "traffic_level": request.traffic_level,
        }
        input_df = pd.DataFrame([input_dict])
        prediction = round(float(model.predict(input_df)[0]), 2)

        # Log every prediction
        log_prediction(input_dict, prediction)

        return PredictionResponse(delivery_time_min=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")