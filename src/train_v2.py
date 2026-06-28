import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH       = "data/delivery_times.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"


def main():
    print("=== QuickFoods: Train and Register Model V2 ===")

    df = pd.read_csv(DATA_PATH)
    X  = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y  = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=99  # different split
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest-V2"):
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("purpose", "version_comparison")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=3,
            random_state=99,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2   = r2_score(y_test, preds)

        mlflow.log_param("model_name", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("min_samples_split", 3)
        mlflow.log_param("random_state", 99)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model in MLflow format
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        # Register as new version
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/sklearn-model"
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R²   : {r2:.3f}")
        print(f"\n✅ Registered '{REGISTERED_NAME}' version {mv.version}")


if __name__ == "__main__":
    main()