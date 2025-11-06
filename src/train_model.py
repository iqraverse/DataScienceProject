# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib
import os

# Optional: MLflow helper if using MLflow tracking
try:
    from mlflow_helpers import init_mlflow, log_model_with_metrics
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False
    print("⚠️ MLflow helpers not found — skipping MLflow logging.")

# 1️⃣ Load training features
print("📂 Loading training features...")
df = pd.read_csv("data/daily/training_features.csv")

# 2️⃣ Define features (X) and target (y)
X = df.drop(columns=["AQI", "time"])
y = df["AQI"]

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train three horizon models (example setup)
print("🚀 Training models...")
model_h1 = RandomForestRegressor(random_state=42)
model_h2 = RandomForestRegressor(random_state=42)
model_h3 = RandomForestRegressor(random_state=42)

model_h1.fit(X_train, y_train)
model_h2.fit(X_train, y_train)
model_h3.fit(X_train, y_train)

# 5️⃣ Evaluate metrics
pred_h1 = model_h1.predict(X_test)
pred_h2 = model_h2.predict(X_test)
pred_h3 = model_h3.predict(X_test)

mae_h1 = mean_absolute_error(y_test, pred_h1)
rmse_h1 = root_mean_squared_error(y_test, pred_h1)

mae_h2 = mean_absolute_error(y_test, pred_h2)
rmse_h2 = root_mean_squared_error(y_test, pred_h2)

mae_h3 = mean_absolute_error(y_test, pred_h3)
rmse_h3 = root_mean_squared_error(y_test, pred_h3)

print(f"📊 H+1 MAE={mae_h1:.2f} RMSE={rmse_h1:.2f}")
print(f"📊 H+2 MAE={mae_h2:.2f} RMSE={rmse_h2:.2f}")
print(f"📊 H+3 MAE={mae_h3:.2f} RMSE={rmse_h3:.2f}")

# 6️⃣ Save trained models
os.makedirs("models", exist_ok=True)
joblib.dump(model_h1, "models/model_h1.joblib")
joblib.dump(model_h2, "models/model_h2.joblib")
joblib.dump(model_h3, "models/model_h3.joblib")
print("💾 Models saved to /models folder.")

# 7️⃣ Log to MLflow (optional)
if USE_MLFLOW:
    init_mlflow()
    metrics_h1 = {"mae": mae_h1, "rmse": rmse_h1}
    metrics_h2 = {"mae": mae_h2, "rmse": rmse_h2}
    metrics_h3 = {"mae": mae_h3, "rmse": rmse_h3}

    input_example = X_train.iloc[:1]

    runid1 = log_model_with_metrics(model_h1, "aqi_model_h1", metrics_h1, sample_input=input_example)
    runid2 = log_model_with_metrics(model_h2, "aqi_model_h2", metrics_h2, sample_input=input_example)
    runid3 = log_model_with_metrics(model_h3, "aqi_model_h3", metrics_h3, sample_input=input_example)


    print("🧠 MLflow run IDs:", runid1, runid2, runid3)
else:
    print("⚙️ Skipping MLflow logging (helpers not imported).")

print("✅ Training complete.")
