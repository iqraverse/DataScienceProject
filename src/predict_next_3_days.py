# src/predict_next_3_days.py
import pandas as pd
import joblib
import os
from datetime import timedelta

print("📈 Starting recursive 3-day AQI forecasting...")

# ✅ Load trained models
model_h1 = joblib.load("models/model_h1.joblib")
model_h2 = joblib.load("models/model_h2.joblib")
model_h3 = joblib.load("models/model_h3.joblib")

# ✅ Load feature dataset used during training
feature_path = "data/daily/training_features.csv"
if not os.path.exists(feature_path):
    raise FileNotFoundError("❌ training_features.csv not found. Run prepareFeatures.py first.")

df = pd.read_csv(feature_path, parse_dates=["time"])
df = df.sort_values("time")

# ✅ Start from the latest available feature row (with lags & rolls)
base_row = df.iloc[-1:].copy()
last_date = df["time"].max()

predictions = []

# --- Forecast 1: +1 day ahead ---------------------------------
pred_1 = model_h1.predict(base_row.drop(columns=["time", "AQI"], errors="ignore"))[0]
predictions.append((last_date + timedelta(days=1), pred_1))

# --- Prepare next input for Forecast 2 -------------------------
row_h2 = base_row.copy()
row_h2["AQI_lag_1"] = pred_1
# shift other lags (lag_2 <- lag_1, etc.) if they exist
for i in range(2, 8):
    if f"AQI_lag_{i}" in row_h2.columns:
        row_h2[f"AQI_lag_{i}"] = base_row[f"AQI_lag_{i-1}"].values[0]
# recompute rolling windows roughly
if "AQI_roll3" in row_h2.columns:
    row_h2["AQI_roll3"] = (pred_1 + base_row["AQI_lag_1"].values[0] + base_row["AQI_lag_2"].values[0]) / 3
if "AQI_roll7" in row_h2.columns:
    row_h2["AQI_roll7"] = (
        pred_1
        + base_row["AQI_lag_1"].values[0]
        + base_row["AQI_lag_2"].values[0]
        + base_row["AQI_lag_3"].values[0]
        + base_row["AQI_lag_4"].values[0]
        + base_row["AQI_lag_5"].values[0]
        + base_row["AQI_lag_6"].values[0]
    ) / 7

pred_2 = model_h2.predict(row_h2.drop(columns=["time", "AQI"], errors="ignore"))[0]
predictions.append((last_date + timedelta(days=2), pred_2))

# --- Prepare next input for Forecast 3 -------------------------
row_h3 = row_h2.copy()
row_h3["AQI_lag_1"] = pred_2
for i in range(2, 8):
    if f"AQI_lag_{i}" in row_h3.columns:
        row_h3[f"AQI_lag_{i}"] = row_h2[f"AQI_lag_{i-1}"].values[0]
if "AQI_roll3" in row_h3.columns:
    row_h3["AQI_roll3"] = (pred_2 + row_h2["AQI_lag_1"].values[0] + row_h2["AQI_lag_2"].values[0]) / 3
if "AQI_roll7" in row_h3.columns:
    row_h3["AQI_roll7"] = (
        pred_2
        + row_h2["AQI_lag_1"].values[0]
        + row_h2["AQI_lag_2"].values[0]
        + row_h2["AQI_lag_3"].values[0]
        + row_h2["AQI_lag_4"].values[0]
        + row_h2["AQI_lag_5"].values[0]
        + row_h2["AQI_lag_6"].values[0]
    ) / 7

pred_3 = model_h3.predict(row_h3.drop(columns=["time", "AQI"], errors="ignore"))[0]
predictions.append((last_date + timedelta(days=3), pred_3))

# ✅ Combine and save results
pred_df = pd.DataFrame(predictions, columns=["date", "predicted_AQI"])
os.makedirs("data/daily", exist_ok=True)
pred_df.to_csv("data/daily/predictions.csv", index=False)

print("✅ Recursive 3-day predictions saved to data/daily/predictions.csv")
print(pred_df)
