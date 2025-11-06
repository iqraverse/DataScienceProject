import pandas as pd
import os

csv_path = "data/daily/clean_air_quality_data.csv"
parquet_path = "data/daily/clean_air_quality_data.parquet"

# ✅ Step 1: Load CSV
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at {csv_path}. Run data_cleaning_check.py first.")

df = pd.read_csv(csv_path)
print(f"📥 Loaded {len(df)} rows from {csv_path}")

# ✅ Step 2: Convert 'time' to timezone-aware datetime (UTC)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    print("🕒 Converted 'time' column to timezone-aware datetime")

# ✅ Step 3: Add city column if missing
if "city" not in df.columns:
    df["city"] = "Karachi"
    print("🏙️ Added 'city' column for Feast entity")

# ✅ Step 4: Save as Parquet file
df.to_parquet(parquet_path, index=False)
print(f"✅ Parquet file created at {parquet_path}")
