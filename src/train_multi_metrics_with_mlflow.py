import os
import json
import time
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    X = df[
        [
            "distance_km",
            "items_count",
            "is_peak_hour",
            "traffic_level",
        ]
    ]

    y = df["delivery_time_min"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def measure_latency(model, sample, repeats=200):
    model.predict(sample)

    start = time.perf_counter()

    for _ in range(repeats):
        model.predict(sample)

    end = time.perf_counter()

    return ((end - start) / repeats) * 1000


def log_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):

        importance = dict(
            zip(
                feature_names,
                model.feature_importances_,
            )
        )

        mlflow.log_dict(
            importance,
            "feature_importance.json",
        )


def train_and_track(
    model_name,
    model,
    params,
    X_train,
    X_test,
    y_train,
    y_test,
):

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("problem_type", "Regression")

        mlflow.log_param("model_name", model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
        )

        cv_mae = abs(cv_scores.mean())

        mlflow.log_metric(
            "cv_mae",
            float(cv_mae),
        )

        ensure_dir(MODEL_DIR)

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl",
        )

        joblib.dump(
            model,
            model_path,
        )

        size_kb = (
            os.path.getsize(model_path)
            / 1024
        )

        mlflow.log_metric(
            "model_size_kb",
            float(size_kb),
        )

        sample = X_test.iloc[[0]]

        latency = measure_latency(
            model,
            sample,
        )

        mlflow.log_metric(
            "avg_latency_ms",
            float(latency),
        )

        mlflow.log_artifact(model_path)

        report = {
            "model": model_name,
            "params": params,
            "metrics": metrics,
            "cv_mae": cv_mae,
            "model_size_kb": size_kb,
            "latency_ms": latency,
        }

        report_path = os.path.join(
            MODEL_DIR,
            f"{model_name}_report.json",
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=4,
            )

        mlflow.log_artifact(report_path)

        log_feature_importance(
            model,
            X_train.columns,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="delivery-time-model",
        )

        print(
            f"[OK] {model_name} | "
            f"MAE={metrics['mae']:.3f} | "
            f"RMSE={metrics['rmse']:.3f} | "
            f"R2={metrics['r2']:.3f} | "
            f"CV_MAE={cv_mae:.3f} | "
            f"Size={size_kb:.1f}KB | "
            f"Latency={latency:.3f}ms"
        )

        return {
            "model_name": model_name,
            **metrics,
            "cv_mae": cv_mae,
            "model_size_kb": size_kb,
            "latency_ms": latency,
            "model": model,
        }


def main():

    print("=== Exercise 03: MLflow Multi-Metric Tracking (QuickFoods) ===")

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(df)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    results = []

    results.append(
        train_and_track(
            "LinearRegression",
            LinearRegression(),
            {},
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results.append(
        train_and_track(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
            ),
            {
                "n_estimators": 150
            },
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results.append(
        train_and_track(
            "GradientBoosting",
            GradientBoostingRegressor(
                random_state=42
            ),
            {},
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    if XGBOOST_AVAILABLE:

        results.append(
            train_and_track(
                "XGBoost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                ),
                {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                },
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    best = min(
        results,
        key=lambda x: x["mae"]
    )

    ensure_dir(MODEL_DIR)

    best_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl",
    )

    joblib.dump(
        best["model"],
        best_path,
    )

    print(
        "\n=== BEST MODEL ==="
    )

    print(
        f"\nModel: {best['model_name']}"
    )

    print(
        f"MAE: {best['mae']:.3f}"
    )

    print(
        f"RMSE: {best['rmse']:.3f}"
    )

    print(
        f"R2: {best['r2']:.3f}"
    )

    print(
        f"CV MAE: {best['cv_mae']:.3f}"
    )

    print(
        f"Saved: {best_path}"
    )

    print(
        "\nRun MLflow UI:"
    )

    print(
        "mlflow ui"
    )

    print(
        "Open: http://127.0.0.1:5000"
    )


if __name__ == "__main__":
    main()
