# src/predict_latest.py
import os
import sys
import pandas as pd
import joblib
from datetime import timedelta

# ✅ Make sure Python can find the 'src' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prepareFeatures import create_lag_features

# --- Load data ---
data_path = "data/daily/clean_air_quality_data.csv"
df = pd.read_csv(data_path)

# Automatically find the date column
date_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

if not date_col:
    raise KeyError("No date or time column found in dataset!")

# --- Create lag features (same as training) ---
df = create_lag_features(data_path, "data/daily/training_features.csv")

# --- Get latest record with lag features ---
latest = df.iloc[-1:].copy()

# --- Create next date ---
next_date = (pd.to_datetime(latest[date_col].iloc[0]) + timedelta(days=1)).strftime("%Y-%m-%d")

# --- Load model ---
model_path = "models/model_h1.joblib"
model = joblib.load(model_path)

# --- Drop non-feature columns ---
drop_cols = [date_col]
if "AQI" in df.columns:
    drop_cols.append("AQI")
features = latest.drop(columns=drop_cols)

# --- Predict ---
predicted_aqi = model.predict(features)[0]

# --- Save to predictions.csv ---
pred_path = "data/daily/predictions.csv"
if os.path.exists(pred_path):
    preds = pd.read_csv(pred_path)
else:
    preds = pd.DataFrame(columns=["date", "predicted_aqi"])

new_row = pd.DataFrame({"date": [next_date], "predicted_aqi": [predicted_aqi]})
preds = pd.concat([preds, new_row], ignore_index=True)
preds.to_csv(pred_path, index=False)

print(f"✅ New prediction added for {next_date}: {predicted_aqi:.2f}")
