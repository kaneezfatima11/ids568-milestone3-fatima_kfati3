import mlflow
from mlflow.tracking import MlflowClient


def validate_latest_run():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        "ids568_milestone3_rf_experiments"
    )

    if experiment is None:
        raise Exception("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise Exception("No runs found")

    latest_run = runs[0]

    accuracy = latest_run.data.metrics.get("accuracy")
    f1 = latest_run.data.metrics.get("f1_score")

    print(f"Latest run accuracy: {accuracy}")
    print(f"Latest run f1_score: {f1}")

    # Simple validation rule
    if accuracy < 0.90:
        raise Exception("Model accuracy below acceptable threshold")

    print("Model validation passed.")