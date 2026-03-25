import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
force_fail = os.getenv("FORCE_FAIL", "false").lower() == "true"

if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow.set_experiment("mlops-assignment")

df = pd.read_csv("data/dataset.csv")

X = df[["feature1", "feature2", "feature3", "feature4"]]
y = df["target"]

if force_fail:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=1, stratify=y
    )
    model = LogisticRegression(
        max_iter=5,
        C=0.001,
        solver="lbfgs"
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs"
    )

with mlflow.start_run() as run:
    run_id = run.info.run_id

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("force_fail", force_fail)
    mlflow.log_metric("accuracy", accuracy)

    print("Run ID:", run_id)
    print("Accuracy:", accuracy)

    with open("model_info.txt", "w", encoding="utf-8") as f:
        f.write(run_id)