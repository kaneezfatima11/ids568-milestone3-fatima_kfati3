from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import mlflow
import hashlib
from mlflow.tracking import MlflowClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from preprocess import preprocess
from train import main as train_main
from model_validation import validate_latest_run


def on_failure_callback(context):
    print(f"Task {context['task_instance'].task_id} failed.")


def register_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ids568_milestone3_rf_experiments")

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
    run_id = latest_run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(model_uri, "BreastCancerRF")

    client.transition_model_version_stage(
        name="BreastCancerRF",
        version=registered_model.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    model_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    client.set_model_version_tag(
        name="BreastCancerRF",
        version=registered_model.version,
        key="model_hash",
        value=model_hash,
    )

    print(f"Model registered as version {registered_model.version} in Staging.")


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="mlflow_training_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=lambda: train_main(120, 6),
    )

    validate_task = PythonOperator(
        task_id="validate_model",
        python_callable=validate_latest_run,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    preprocess_task >> train_task >> validate_task >> register_task