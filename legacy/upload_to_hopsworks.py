import os
import pandas as pd
import hopsworks

# ✅ Upload training feature set (AQI + lags + rolling + time features)
IN_FILE = "data/daily/training_features.csv"

def upload():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "open_meteo")

    if not api_key:
        raise RuntimeError("🚨 Missing HOPSWORKS_API_KEY. Please set it in your .env file or GitHub Secrets.")

    print("🔐 Logging in to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    print(f"📥 Loading file: {IN_FILE}")
    df = pd.read_csv(IN_FILE, parse_dates=["time"])
    df = df.sort_values("time")

    # ✅ Extra metadata for future scaling
    df["city"] = "Karachi"

    # ✅ Convert 'time' column to string for primary key (Hopsworks requirement)
    df["time"] = df["time"].astype(str)

    # ✅ Create 'event_time' column (as timestamp)
    df["event_time"] = pd.to_datetime(df["time"])

    # ✅ Create or get feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_features_daily",
        version=1,
        primary_key=["time", "city"],
        event_time="event_time",
        description="Prepared AQI training features for Karachi",
        online_enabled=True
    )

    print(f"📤 Inserting {len(df)} rows into Feature Store...")
    fg.insert(df, write_options={"wait_for_job": True})

    print("✅ Feature Store Upload Successful!")
    print("🌟 Feature Group: aqi_features_daily v1 created/updated")

if __name__ == "__main__":
    upload()
