import argparse
import json
import os
import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "models/delivery_time_model.pkl")

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}. Train a model first and save it to {path}."
        )
    return joblib.load(path)

def predict_one(model, distance_km: float, items_count: int, is_peak_hour: int, traffic_level: int) -> float:
    X = pd.DataFrame([{
        "distance_km": distance_km,
        "items_count": items_count,
        "is_peak_hour": is_peak_hour,
        "traffic_level": traffic_level
    }])
    pred = model.predict(X)[0]
    return float(pred)

def main():
    parser = argparse.ArgumentParser(description="FooFoods Delivery Time Predictor (CLI)")
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--items_count", type=int, required=True)
    parser.add_argument("--is_peak_hour", type=int, choices=[0, 1], required=True)
    parser.add_argument("--traffic_level", type=int, choices=[1, 2, 3], required=True)

    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    y = predict_one(model, args.distance_km, args.items_count, args.is_peak_hour, args.traffic_level)

    out = {
        "input": {
            "distance_km": args.distance_km,
            "items_count": args.items_count,
            "is_peak_hour": args.is_peak_hour,
            "traffic_level": args.traffic_level
        },
        "prediction": {
            "delivery_time_min": round(y, 2)
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()