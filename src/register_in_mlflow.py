# src/register_in_mlflow.py
import mlflow
from mlflow.tracking import MlflowClient

# ✅ Local MLflow tracking directory
mlflow.set_tracking_uri("mlruns")

# ✅ Experiment name (same as training)
experiment_name = "AQI_Predictor_With_FeatureStore"

# ✅ Registered model name
registered_model_name = "AQI_Model_FeatureStore"

client = MlflowClient()

# 1️⃣ Get last run ID from your experiment
experiment = client.get_experiment_by_name(experiment_name)
if not experiment:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
if not runs:
    raise ValueError("No runs found for this experiment.")

last_run = runs[0]
run_id = last_run.info.run_id

print(f"📦 Latest run ID: {run_id}")

# 2️⃣ Register the model
model_uri = f"runs:/{run_id}/model"
try:
    client.create_registered_model(registered_model_name)
    print(f"✅ Created new registered model: {registered_model_name}")
except Exception:
    print(f"ℹ️ Model '{registered_model_name}' already exists. Registering new version...")

model_version = client.create_model_version(
    name=registered_model_name,
    source=model_uri,
    run_id=run_id,
)

# 3️⃣ Transition to "Staging"
client.transition_model_version_stage(
    name=registered_model_name,
    version=model_version.version,
    stage="Staging"
)

print(f"✅ Registered and moved model v{model_version.version} to 'Staging' stage!")
