# src/train_with_feast.py
import os
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow, mlflow.sklearn
import joblib

# ---------- config ----------
FEAST_REPO = "feast_repo"
PARQUET_PATH = "data/daily/clean_air_quality_data.parquet"
MLFLOW_LOCAL_PATH = "mlruns"
MODEL_OUT = "models/model_from_feast.joblib"
EXPERIMENT_NAME = "AQI_Predictor_With_FeatureStore"
# ----------------------------

print("📥 Initializing Feast FeatureStore...")
fs = FeatureStore(repo_path=FEAST_REPO)

# --- Build entity_df: must include entity keys and column named `event_timestamp`
print("📥 Loading entity dataframe from parquet:", PARQUET_PATH)
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"{PARQUET_PATH} not found. Convert CSV -> parquet first.")

entity_df = pd.read_parquet(PARQUET_PATH, columns=["city", "time"])
entity_df = entity_df.rename(columns={"time": "event_timestamp"})
entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
print("✅ entity_df shape:", entity_df.shape)

# ---- Features list ----
feature_list = [
    "aqi_features:pm10",
    "aqi_features:pm2_5",
    "aqi_features:carbon_monoxide",
    "aqi_features:nitrogen_dioxide",
    "aqi_features:sulphur_dioxide",
    "aqi_features:ozone",
    "aqi_features:AQI",
]

print("📥 Retrieving training data from Feast (historical join)...")
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=feature_list,
).to_df()

print("✅ Retrieved Feast training data:", training_df.shape)
training_df = training_df.dropna()

if "AQI" not in training_df.columns:
    raise RuntimeError("❌ Target column 'AQI' not found in historical retrieval result.")

# ---- Split features / target ----
X = training_df.drop(columns=["AQI", "event_timestamp", "city"])
y = training_df["AQI"]

print("📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- MLflow setup ----
os.makedirs(MLFLOW_LOCAL_PATH, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_LOCAL_PATH)
mlflow.set_experiment(EXPERIMENT_NAME)
print("✅ MLflow tracking:", os.path.abspath(MLFLOW_LOCAL_PATH))

# ---- Train & log model ----
print("🚀 Training model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

with mlflow.start_run(run_name="feast_rf_model"):
    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("rmse", float(rmse))
    input_example = X_train.iloc[:1]
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

print(f"✅ Model trained & logged | MAE={mae:.3f}, RMSE={rmse:.3f}")

# ---- Save model locally ----
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)
print("💾 Model saved to", MODEL_OUT)
