import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
# ── Config ─────────────────────────────────────────────────────────────────────

ORIGINAL_DATA   = "data/delivery_times.csv"
NEW_DATA        = "data/delivery_times_new.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"
CHAMPION_ALIAS  = "champion"

FEATURES = ["distance_km", "items_count", "is_peak_hour", "traffic_level"]
TARGET   = "delivery_time_min"


def load_champion_model():
    """Load the current champion from the registry."""
    model_uri = f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}"
    print(f"Loading current champion from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def main():
    print("=== QuickFoods: Retraining Pipeline ===\n")

    # ── Load and combine data ─────────────────────────────────────────────
    df_old = pd.read_csv(ORIGINAL_DATA)
    df_new = pd.read_csv(NEW_DATA)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    print(f"Original data : {len(df_old)} rows")
    print(f"New data      : {len(df_new)} rows")
    print(f"Combined      : {len(df_combined)} rows")

    X = df_combined[FEATURES]
    y = df_combined[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Evaluate current champion on new test set ─────────────────────────
    champion_model = load_champion_model()
    champion_preds = champion_model.predict(X_test)
    champion_mae   = mean_absolute_error(y_test, champion_preds)
    champion_rmse  = float(np.sqrt(mean_squared_error(y_test, champion_preds)))
    champion_r2    = r2_score(y_test, champion_preds)

    print(f"\nCurrent Champion on new test set:")
    print(f"  MAE  : {champion_mae:.3f}")
    print(f"  RMSE : {champion_rmse:.3f}")
    print(f"  R²   : {champion_r2:.3f}")

    # ── Train new model on combined data ──────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Retrain-CombinedData"):
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("purpose", "retraining")
        mlflow.set_tag("data", "original+new")

        new_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        new_model.fit(X_train, y_train)

        new_preds = new_model.predict(X_test)
        new_mae   = mean_absolute_error(y_test, new_preds)
        new_rmse  = float(np.sqrt(mean_squared_error(y_test, new_preds)))
        new_r2    = r2_score(y_test, new_preds)

        mlflow.log_param("model_name", "GradientBoosting")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("data_rows", len(df_combined))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("mae", new_mae)
        mlflow.log_metric("rmse", new_rmse)
        mlflow.log_metric("r2", new_r2)

        # Also log champion metrics for comparison
        mlflow.log_metric("champion_mae", champion_mae)
        mlflow.log_metric("champion_rmse", champion_rmse)
        mlflow.log_metric("champion_r2", champion_r2)
        mlflow.log_metric("mae_improvement", champion_mae - new_mae)

        mlflow.sklearn.log_model(new_model, artifact_path="sklearn-model")

        run_id = mlflow.active_run().info.run_id

        print(f"\nRetrained Model on new test set:")
        print(f"  MAE  : {new_mae:.3f}")
        print(f"  RMSE : {new_rmse:.3f}")
        print(f"  R²   : {new_r2:.3f}")

        # ── Compare and decide ────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Champion MAE : {champion_mae:.3f}")
        print(f"  Retrained MAE: {new_mae:.3f}")
        print(f"  Improvement  : {champion_mae - new_mae:.3f} minutes")
        print(f"{'='*50}")

        if new_mae < champion_mae:
            print("\n✅ Retrained model is BETTER. Registering and promoting...")

            model_uri = f"runs:/{run_id}/sklearn-model"
            mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.set_registered_model_alias(REGISTERED_NAME, CHAMPION_ALIAS, mv.version)

            print(f"   Registered version {mv.version}")
            print(f"   Promoted to '{CHAMPION_ALIAS}' alias")
            print(f"\n   Restart the FastAPI service to serve the new model.")
        else:
            print("\n❌ Retrained model is NOT better. Keeping current champion.")
            print("   Consider: more data, different features, or different algorithm.")


if __name__ == "__main__":
    main()