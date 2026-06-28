import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "quickfoods-delivery-time"
REGISTERED_NAME = "quickfoods-delivery-predictor"
METRIC          = "mae"

def main():
    print("=== QuickFoods: Promote Best Tuned Model to Registry ===")

    client     = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train_hyperparameter_tuning.py first.")

    # Fetch all runs ordered by MAE — includes child runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC} > 0",
        order_by=[f"metrics.{METRIC} ASC"],
        max_results=100
    )

    # Only child runs have the model artifact we want
    candidates = [r for r in runs if r.data.tags.get("sweep_child") == "true"]

    if not candidates:
        raise ValueError("No child trial runs found. Check that sweep_child tag is set in train_hyperparameter_tuning.py.")

    best     = candidates[0]
    best_mae = best.data.metrics[METRIC]

    print(f"Best run ID : {best.info.run_id}")
    print(f"Best MAE    : {best_mae:.3f} minutes")
    print(f"Model       : {best.data.params.get('model_name')}")
    print(f"Params      : { {k: v for k, v in best.data.params.items() if k != 'model_name'} }")

    model_uri = f"runs:/{best.info.run_id}/sklearn-model"
    mv        = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

    print(f"\n✅ Registered '{REGISTERED_NAME}'  version {mv.version}")
    print(f"   Status: {mv.status}")
    print(f"\nView in MLflow UI → Models tab → {REGISTERED_NAME}")


if __name__ == "__main__":
    main()