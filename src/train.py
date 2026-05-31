import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "delivery_time_model.pkl")

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)

def train_model(df: pd.DataFrame) -> dict:
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "test_size": len(X_test),
    }

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    print("=== QuickFoods MLOps Lab 1: Baseline Training ===")

    df = load_data(DATA_PATH)
    result = train_model(df)

    print(f"Test samples: {result['test_size']}")
    print(f"MAE (minutes): {result['mae']:.2f}")
    print(f"MSE: {result['mse']:.2f}")

    save_model(result["model"])
    print(f"Model saved to: {MODEL_PATH}")
    print("Done. Reproducible ML artifact created.")

if __name__ == "__main__":
    main()
