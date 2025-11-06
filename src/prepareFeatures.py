# src/prepareFeatures.py
import pandas as pd
import numpy as np
import os

def create_lag_features(input_path="data/daily/clean_air_quality_data.csv",
                        output_path="data/daily/training_features.csv"):
    """
    Reads cleaned AQI data, creates lag and rolling features,
    and saves them for model training or prediction.
    """
    print("📥 Loading cleaned data...")
    df = pd.read_csv(input_path, parse_dates=["time"])
    df = df.sort_values("time")

    # Generate lag features
    for lag in range(1, 8):
        df[f"AQI_lag_{lag}"] = df["AQI"].shift(lag)

    # Rolling mean features
    df["AQI_roll3"] = df["AQI"].rolling(window=3).mean()
    df["AQI_roll7"] = df["AQI"].rolling(window=7).mean()

    # Drop NA rows created by lagging
    df = df.dropna().reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Features prepared and saved to {output_path}")
    print(df.head())

    return df  # ✅ so other files can reuse it
