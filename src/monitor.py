import json
import os
import pandas as pd
import numpy as np

LOG_PATH      = "logs/predictions.jsonl"
TRAINING_DATA = "data/delivery_times.csv"

# Alert thresholds (business-defined)
ALERT_DISTANCE_MEAN_SHIFT = 3.0    # if mean distance shifts by more than 3 km
ALERT_ITEMS_MEAN_SHIFT    = 2.0    # if mean items shifts by more than 2
ALERT_PREDICTION_MEAN     = 60.0   # if avg prediction exceeds 60 min, something is off


def load_prediction_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No prediction log at: {path}")

    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    rows = []
    for r in records:
        row = {
            "timestamp": r["timestamp"],
            "prediction": r["prediction"],
            **r["input"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_prediction_stats(df: pd.DataFrame) -> dict:
    return {
        "total_predictions": len(df),
        "mean_prediction": round(df["prediction"].mean(), 2),
        "max_prediction": round(df["prediction"].max(), 2),
        "min_prediction": round(df["prediction"].min(), 2),
        "std_prediction": round(df["prediction"].std(), 2),
    }


def check_input_drift(live_df: pd.DataFrame, train_df: pd.DataFrame) -> list:
    alerts = []

    # Compare mean distance
    train_mean_dist = train_df["distance_km"].mean()
    live_mean_dist  = live_df["distance_km"].mean()
    dist_shift      = abs(live_mean_dist - train_mean_dist)

    if dist_shift > ALERT_DISTANCE_MEAN_SHIFT:
        alerts.append(
            f"DRIFT: distance_km mean shifted by {dist_shift:.1f} km "
            f"(train={train_mean_dist:.1f}, live={live_mean_dist:.1f})"
        )

    # Compare mean items
    train_mean_items = train_df["items_count"].mean()
    live_mean_items  = live_df["items_count"].mean()
    items_shift      = abs(live_mean_items - train_mean_items)

    if items_shift > ALERT_ITEMS_MEAN_SHIFT:
        alerts.append(
            f"DRIFT: items_count mean shifted by {items_shift:.1f} "
            f"(train={train_mean_items:.1f}, live={live_mean_items:.1f})"
        )

    # Check prediction range
    mean_pred = live_df["prediction"].mean()
    if mean_pred > ALERT_PREDICTION_MEAN:
        alerts.append(
            f"WARNING: avg prediction is {mean_pred:.1f} min — exceeds threshold of {ALERT_PREDICTION_MEAN} min"
        )

    return alerts


def main():
    print("=== QuickFoods Model Monitoring Report ===\n")

    live_df  = load_prediction_log(LOG_PATH)
    train_df = pd.read_csv(TRAINING_DATA)

    # Prediction statistics
    stats = compute_prediction_stats(live_df)
    print("Prediction Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Feature distribution summary
    print("\nLive Feature Means:")
    for col in ["distance_km", "items_count", "is_peak_hour", "traffic_level"]:
        print(f"  {col}: {live_df[col].mean():.2f}  (train: {train_df[col].mean():.2f})")

    # Drift alerts
    print("\nDrift Check:")
    alerts = check_input_drift(live_df, train_df)

    if alerts:
        for a in alerts:
            print(f"  ⚠️  {a}")
        print("\n→ Action: Investigate data source. Consider retraining (Exercise 09).")
    else:
        print("  ✅ No drift detected. Model inputs look consistent with training data.")

    print("\nDone.")


if __name__ == "__main__":
    main()