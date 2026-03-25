import os
import sys
import mlflow

THRESHOLD = 0.85

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

if not os.path.exists("model_info.txt"):
    print("model_info.txt not found")
    sys.exit(1)

with open("model_info.txt") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

print("Accuracy:", accuracy)

if accuracy < THRESHOLD:
    print("Accuracy below threshold")
    sys.exit(1)

print("Accuracy passed threshold")
