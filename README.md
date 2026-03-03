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

**Conclusion**
This project implements a production-style ML workflow combining:
Airflow orchestration
MLflow experiment tracking
Reproducible model training
Fully automated pipeline.