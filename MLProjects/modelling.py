import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Setup MLflow
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
experiment = mlflow.set_experiment("Shipping-Delay")

def main(data_folder):
    # Load data
    X_train = pd.read_csv(f"{data_folder}/X_train.csv")
    X_test = pd.read_csv(f"{data_folder}/X_test.csv")
    y_train = pd.read_csv(f"{data_folder}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_folder}/y_test.csv").values.ravel()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()
        
        model = XGBClassifier()
        model.fit(X_train, y_train)
        
        # Log model secara eksplisit
        mlflow.sklearn.log_model(model, "model")
        print(f"Model saved to: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    args = parser.parse_args()
    main(args.data_folder)