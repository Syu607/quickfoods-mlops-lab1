import os
import json
import mlflow
import pandas as pd
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


REGISTERED_NAME = "quickfoods-delivery-predictor"
CHAMPION_ALIAS  = "champion"
LOG_DIR         = "logs"
LOG_PATH        = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="QuickFoods Delivery Time Prediction API",
    description="Serves the champion model from MLflow Registry",
    version="3.0.0"
)

os.makedirs(LOG_DIR, exist_ok=True)

# Load the champion model from the registry
model_uri = f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}"
print(f"Loading model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully.")


class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)


class PredictionResponse(BaseModel):
    delivery_time_min: float


def log_prediction(request_data: dict, prediction: float):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_data,
        "prediction": prediction,
        "model": REGISTERED_NAME,
        "alias": CHAMPION_ALIAS,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": REGISTERED_NAME,
        "alias": CHAMPION_ALIAS,
        "model_uri": model_uri,
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

        log_prediction(input_dict, prediction)

        return PredictionResponse(delivery_time_min=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")