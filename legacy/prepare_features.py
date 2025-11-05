import pandas as pd

def prepare_features(input_file="data/daily/clean_air_quality_data.csv",
                     output_file="data/daily/training_features.csv"):
    df = pd.read_csv(input_file, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Time features
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Lag features for AQI (past 7 days)
    for lag in range(1, 8):
        df[f"AQI_lag_{lag}"] = df["AQI"].shift(lag)

    # Rolling averages
    df["AQI_roll3"] = df["AQI"].rolling(window=3).mean()
    df["AQI_roll7"] = df["AQI"].rolling(window=7).mean()

    df = df.dropna()  # remove incomplete rows

    df.to_csv(output_file, index=False)
    print(f"✅ Features prepared and saved: {output_file}")
    print(df.head())

if __name__ == "__main__":
    prepare_features()
