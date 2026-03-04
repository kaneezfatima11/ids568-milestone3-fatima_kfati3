# IDS 568 – Milestone 3  
## MLflow + Airflow Orchestrated Machine Learning Pipeline

------------------------------------------------------------

PROJECT OVERVIEW

This project demonstrates an end-to-end MLOps workflow using:

- Apache Airflow (pipeline orchestration)
- MLflow (experiment tracking)
- Scikit-learn (model training)
- Random Forest classifier
- Breast Cancer dataset

The pipeline includes:
1. Data preprocessing
2. Train/test split generation
3. Model training
4. MLflow experiment tracking
5. Artifact logging
6. DAG-based orchestration via Airflow

------------------------------------------------------------

PROJECT STRUCTURE

ids568-milestone3-fatima_kfati3/
│
├── preprocess.py          (Data preprocessing & split generation)
├── train.py               (Model training + MLflow logging)
├── artifacts/             (Saved train/test splits & model.pkl)
├── requirements.txt       (Python dependencies)
├── .gitignore
└── README.md

Airflow DAG file is located in:
~/airflow/dags/train_pipeline.py

------------------------------------------------------------

ENVIRONMENT SETUP

1) Create Virtual Environment

python3.11 -m venv venv
source venv/bin/activate

2) Install Dependencies
pip install -r requirements.txt

3) Running MLflow
mlflow ui --port 5000
http://localhost:5000

MLflow logs:
1.Model type
2.Hyperparameters
3.Dataset sizes
4.Feature count
5.Accuracy
6.F1 Score
7.Model artifact

4) Running Airflow
airflow standalone
http://localhost:8080

5) Triggering the Pipeline
In Airflow UI:
Toggle the DAG ON
Click Trigger DAG
View Graph to monitor execution

Pipeline order:
preprocess_data → set_tracking_uri → train_model

6) MLflow Logging Details
Each run logs:

Parameters:
model_type
n_estimators
max_depth
train_size
test_size
num_features
data_source
Metrics:
accuracy
f1_score

Artifacts:
model.pkl

**Notes**
Data splits are generated in preprocess.py
train.py loads preprocessed data from artifacts/
Absolute paths are used to ensure Airflow compatibility
MLflow experiment name: ids568_milestone3_rf_experiments
**Notes**
Data splits are generated in preprocess.py
train.py loads preprocessed data from artifacts/
Absolute paths are used to ensure Airflow compatibility
MLflow experiment name: ids568_milestone3_rf_experiments


------------------------------
ARCHITECTURE EXPLANATION
------------------------------

This pipeline uses a layered MLOps architecture integrating Airflow,
MLflow, and GitHub Actions.

Airflow
- Orchestrates the pipeline using a Directed Acyclic Graph (DAG)
- Manages dependencies between preprocessing, training, validation, and model registration

MLflow
- Tracks experiments, parameters, metrics, and artifacts
- Maintains model lineage using the MLflow Model Registry

GitHub Actions
- Runs CI/CD pipelines on code push
- Automatically executes training and validation checks

This architecture ensures reproducibility, traceability, and automated governance of models.


------------------------------
DAG IDEMPOTENCY AND LINEAGE
------------------------------

The Airflow DAG is designed to be idempotent.

Key guarantees:
- Each run produces reproducible outputs
- Data artifacts are written to the artifacts directory
- Model versions are registered with MLflow

Lineage is maintained through MLflow experiment tracking, which records:
- Parameters
- Metrics
- Model artifacts
- Run identifiers

This ensures full traceability from dataset to trained model.


------------------------------
CI-BASED MODEL GOVERNANCE
------------------------------

GitHub Actions enforces model quality through CI pipelines.

The workflow performs:
- Dependency installation
- Dataset artifact generation
- Model training
- Validation checks

If validation fails, the CI job fails and the model is not promoted.
This prevents low-quality models from entering the registry.


------------------------------
EXPERIMENT TRACKING METHODOLOGY
------------------------------

MLflow is used to track experiments and maintain reproducibility.

Each training run logs:

Parameters
- model_type
- n_estimators
- max_depth
- train_size
- test_size
- num_features

Metrics
- accuracy
- f1_score

Artifacts
- trained model.pkl

This allows comparison of multiple runs and model configurations.


------------------------------
RETRY AND FAILURE HANDLING
------------------------------

Airflow tasks include retry mechanisms to improve reliability.

Pipeline configuration includes:
- retries = 2
- retry_delay between attempts
- on_failure_callback for logging failures

This ensures temporary failures (e.g., resource issues) do not break the pipeline.


------------------------------
MONITORING AND ALERTING
------------------------------

Pipeline execution can be monitored using:

Airflow UI
- DAG graph view
- Task status monitoring
- Execution logs

MLflow UI
- Experiment runs
- Metrics and parameters
- Model artifacts

These tools provide visibility into pipeline health and model performance.


------------------------------
ROLLBACK PROCEDURE
------------------------------

If a model performs poorly after deployment:

1. Identify previous stable model version in MLflow Model Registry
2. Transition that version back to Staging or Production
3. Disable or archive the faulty model version

This enables safe rollback to a previously validated model.

**Conclusion**
This project implements a production-style ML workflow combining:
Airflow orchestration
MLflow experiment tracking
Reproducible model training

Fully automated pipeline.

## How to Run the Pipeline

### Run Locally

1. Activate the virtual environment

```
source venv/bin/activate
```

2. Start MLflow UI

```
mlflow ui
```

3. Start Airflow

```
airflow standalone
```

4. Trigger the pipeline

```
airflow dags trigger mlflow_training_pipeline
```

### CI/CD Pipeline

A GitHub Actions workflow automatically runs when code is pushed to the `main` branch.
The workflow performs the following steps:

* Installs dependencies
* Generates dataset artifacts
* Runs model training
* Executes model validation

This ensures the model training pipeline is continuously tested and validated.

