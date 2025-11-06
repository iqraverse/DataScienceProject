# src/mlflow_helpers.py
import mlflow
import mlflow.sklearn
import os
import pandas as pd


def init_mlflow(local_path="mlruns"):
    """
    Initialize MLflow with a Windows-safe local tracking URI.
    """
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
    mlflow.set_tracking_uri(local_path)
    print(f"✅ MLflow tracking URI set to: {os.path.abspath(local_path)}")


def log_model_with_metrics(model, model_name, metrics: dict, artifact_path="model", sample_input=None):
    """
    Log model and metrics to MLflow (new API: name= instead of artifact_path).
    Automatically adds input_example to avoid signature warning.
    """
    init_mlflow()

    # ✅ Ensure experiment always exists (avoids ID 0 error)
    mlflow.set_experiment("AQI_Predictor")


    # ✅ Create dummy sample input if not provided
    if sample_input is None:
        sample_input = pd.DataFrame([{}])

    with mlflow.start_run(run_name=model_name) as run:
        # ✅ Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # ✅ Log model (updated argument)
        mlflow.sklearn.log_model(model, name=artifact_path, input_example=sample_input)

        print(f"✅ Logged {model_name} to MLflow | Run ID: {run.info.run_id}")
        return run.info.run_id
