import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os


def preprocess():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Save splits
    X_train.to_csv("artifacts/X_train.csv", index=False)
    X_test.to_csv("artifacts/X_test.csv", index=False)
    y_train.to_csv("artifacts/y_train.csv", index=False)
    y_test.to_csv("artifacts/y_test.csv", index=False)

    print("Preprocessing complete. Data saved to artifacts/")


if __name__ == "__main__":
    preprocess()