import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import mlflow
import os

from itertools import product as cartesian_product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

mlflow.set_tracking_uri("file:./mlruns")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH       = "data/delivery_times.csv"
MODEL_DIR       = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

FEATURES = ["distance_km", "items_count", "is_peak_hour", "traffic_level"]
TARGET   = "delivery_time_min"

# ── Hyperparameter Grids ───────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [5, 10, None],
    "min_samples_split": [2, 5],
}

GB_PARAM_GRID = {
    "n_estimators":  [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def evaluate(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "r2": float(r2)}


# ── Child Run: one hyperparameter trial ───────────────────────────────────────

def run_trial(model_name, model, params, X_train, X_test, y_train, y_test):
    child_name = model_name + " | " + " ".join(f"{k}={v}" for k, v in params.items())

    with mlflow.start_run(run_name=child_name, nested=True):
        mlflow.set_tag("project",     "QuickFoods")
        mlflow.set_tag("sweep_child", "true")
        mlflow.log_param("model_name", model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        # Cross-validation MAE on the training fold
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring="neg_mean_absolute_error"
        )
        cv_mae = float(-cv_scores.mean())

        mlflow.log_metric("mae",    metrics["mae"])
        mlflow.log_metric("mse",    metrics["mse"])
        mlflow.log_metric("rmse",   metrics["rmse"])
        mlflow.log_metric("r2",     metrics["r2"])
        mlflow.log_metric("cv_mae", cv_mae)

        # Save model artifact
        os.makedirs(MODEL_DIR, exist_ok=True)
        safe_name  = child_name.replace(" | ", "_").replace("=", "").replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        run_id = mlflow.active_run().info.run_id

        print(
            f"  [{model_name}] {params}"
            f" | MAE={metrics['mae']:.3f} | CV_MAE={cv_mae:.3f} | R2={metrics['r2']:.3f}"
        )

        return run_id, metrics["mae"]


# ── Grid Search: RandomForest ──────────────────────────────────────────────────

def grid_search_rf(X_train, X_test, y_train, y_test):
    print("\n=== Grid Search: RandomForestRegressor ===")
    results = []
    keys    = list(RF_PARAM_GRID.keys())
    values  = list(RF_PARAM_GRID.values())

    for combo in cartesian_product(*values):
        params = dict(zip(keys, combo))
        model  = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        run_id, mae = run_trial(
            "RandomForest", model, params,
            X_train, X_test, y_train, y_test
        )
        results.append({"run_id": run_id, "mae": mae, "params": params, "model": "RandomForest"})

    return results


# ── Random Search: GradientBoosting ───────────────────────────────────────────

def random_search_gb(X_train, X_test, y_train, y_test, n_iter=6):
    print(f"\n=== Random Search: GradientBoostingRegressor  (n_iter={n_iter}) ===")
    rng     = np.random.RandomState(RANDOM_STATE)
    results = []

    for _ in range(n_iter):
        params = {k: rng.choice(v).item() for k, v in GB_PARAM_GRID.items()}
        model  = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        run_id, mae = run_trial(
            "GradientBoosting", model, params,
            X_train, X_test, y_train, y_test
        )
        results.append({"run_id": run_id, "mae": mae, "params": params, "model": "GradientBoosting"})

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== QuickFoods MLOps Lab 5: Hyperparameter Tuning ===")

    df = load_data(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="HyperparamSweep-QuickFoods") as parent:
        mlflow.set_tag("project",    "QuickFoods")
        mlflow.set_tag("sweep_type", "grid+random")
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("cv_folds",     3)
        mlflow.log_param("random_state", RANDOM_STATE)

        rf_results = grid_search_rf(X_train, X_test, y_train, y_test)
        gb_results = random_search_gb(X_train, X_test, y_train, y_test, n_iter=6)

        all_results = rf_results + gb_results
        best        = min(all_results, key=lambda r: r["mae"])

        mlflow.log_metric("best_mae",   best["mae"])
        mlflow.log_param("best_run_id", best["run_id"])
        mlflow.log_param("best_model",  best["model"])
        mlflow.log_param("best_params", str(best["params"]))

        print(f"\n{'='*60}")
        print(f"Best model : {best['model']}")
        print(f"Best params: {best['params']}")
        print(f"Best MAE   : {best['mae']:.3f} minutes")
        print(f"Run ID     : {best['run_id']}")
        print(f"Parent ID  : {parent.info.run_id}")
        print(f"{'='*60}")
        print("\nNext: mlflow ui  →  open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()