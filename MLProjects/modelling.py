import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Argument untuk jalur folder dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="ecommerce_shipping_data_preprocessed")
args = parser.parse_args()

# Load dataset dari folder inputan
X_train = pd.read_csv(f"{args.data_folder}/X_train.csv")
X_test = pd.read_csv(f"{args.data_folder}/X_test.csv")
y_train = pd.read_csv(f"{args.data_folder}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{args.data_folder}/y_test.csv").values.ravel()

# Tracking MLflow lokal
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("Shipping Delay Prediction")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Akurasi: {acc}")

    # Simpan model ke direktori lokal
    mlflow.sklearn.log_model(model, "model")