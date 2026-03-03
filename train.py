import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

PROJECT_ROOT = os.path.expanduser("~/ids568-milestone3-fatima_kfati3")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def main(n_estimators, max_depth):
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_test.csv")).values.ravel()

    mlflow.set_experiment("ids568_milestone3_rf_experiments")

    with mlflow.start_run(run_name=f"rf_{n_estimators}_depth_{max_depth}"):

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", "artifacts_preprocessed_split")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Save model artifact
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)

    args = parser.parse_args()

    main(args.n_estimators, args.max_depth)