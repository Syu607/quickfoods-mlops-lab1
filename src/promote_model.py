from mlflow.tracking import MlflowClient

REGISTERED_NAME = "quickfoods-delivery-predictor"


def list_versions(client):
    """Print all versions of the registered model."""
    print(f"\nAll versions of '{REGISTERED_NAME}':")
    # search_model_versions returns all versions
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    for v in versions:
        aliases = v.aliases if hasattr(v, "aliases") else []
        print(f"  Version {v.version} | Run ID: {v.run_id[:8]}... | Aliases: {aliases}")
    return versions


def main():
    print("=== QuickFoods: Model Promotion Workflow ===")

    client = MlflowClient()

    versions = list_versions(client)

    if len(versions) < 2:
        print("\nNeed at least 2 versions. Run train_v2.py first.")
        return

    # Assign version 1 as champion (current production)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", "1")
    print("\n→ Set version 1 as 'champion' (current production model)")

    # Assign version 2 as challenger (candidate for promotion)
    client.set_registered_model_alias(REGISTERED_NAME, "challenger", "2")
    print("→ Set version 2 as 'challenger' (candidate under evaluation)")

    list_versions(client)

    print("\n--- Simulating promotion after evaluation ---\n")

    # After testing: promote challenger to champion
    client.set_registered_model_alias(REGISTERED_NAME, "champion", "2")
    print("→ Promoted version 2 to 'champion'")

    # Remove challenger alias (version 2 is now champion, no longer challenger)
    client.delete_registered_model_alias(REGISTERED_NAME, "challenger")
    print("→ Removed 'challenger' alias from version 2")

    list_versions(client)

    print("\nThe FastAPI service can now load the 'champion' alias to always get the current best model.")
    print("No redeployment needed — just restart the service or implement hot-reload.")


if __name__ == "__main__":
    main()