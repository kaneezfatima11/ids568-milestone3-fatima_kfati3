from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


default_args = {
    "owner": "fatima",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="mlflow_training_pipeline",
    default_args=default_args,
    description="Run MLflow Random Forest training",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "milestone3"],
) as dag:

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="""
        source ~/ids568-milestone3-fatima_kfati3/venv/bin/activate &&
        python ~/ids568-milestone3-fatima_kfati3/preprocess.py
        """,
    )

    set_tracking_uri = BashOperator(
        task_id="set_tracking_uri",
        bash_command="""
        export MLFLOW_TRACKING_URI=http://localhost:5000
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
        source ~/ids568-milestone3-fatima_kfati3/venv/bin/activate &&
        export MLFLOW_TRACKING_URI=http://localhost:5000 &&
        python ~/ids568-milestone3-fatima_kfati3/train.py --n_estimators 120 --max_depth 6
        """,
    )

    preprocess_data >> set_tracking_uri >> train_model