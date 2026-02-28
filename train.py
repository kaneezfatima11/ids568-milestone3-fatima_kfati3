import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os


def main(n_estimators, max_depth):
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    mlflow.set_experiment("ids568_milestone3_rf_experiments")
    with mlflow.start_run(run_name=f"rf_{n_estimators}_depth_{max_depth}"):

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Save model artifact
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
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