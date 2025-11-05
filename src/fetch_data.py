import requests
import pandas as pd
from datetime import date
import os

LAT, LON = 24.8607, 67.001
POLLUTANTS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"

START_DATE = "2024-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

API_URL = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}&hourly={POLLUTANTS}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
)

OUTPUT_FILE = "data/raw/air_quality_data.csv"
os.makedirs("data/raw", exist_ok=True)

def fetch_air_quality_data():
    print(f"🔄 Fetching air quality data up to {END_DATE} ...")

    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    hourly = data.get("hourly", {})

    df_new = pd.DataFrame(hourly)
    if df_new.empty:
        print("⚠️ No new data fetched!")
        return

    # ✅ Convert timestamp properly
    df_new["time"] = pd.to_datetime(df_new["time"], errors="coerce")
    df_new = df_new.dropna(subset=["time"])

    # ✅ Ensure CO stays in µg/m³ (Open-Meteo default)
    if "carbon_monoxide" in df_new.columns and df_new["carbon_monoxide"].notna().any():
        # If mistakenly small values (already mg/m³), convert back to µg/m³
        if df_new["carbon_monoxide"].max() < 100:
            print("🔁 Fixing CO units (mg/m³ → µg/m³)")
            df_new["carbon_monoxide"] = df_new["carbon_monoxide"] * 1000

    # ✅ Sort chronologically (oldest → newest)
    df_new = df_new.sort_values("time", ascending=True)

    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE, parse_dates=["time"])
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["time"], keep="first")
        df_combined = df_combined.sort_values("time", ascending=True)
        new_rows = len(df_combined) - len(df_old)
    else:
        df_combined = df_new
        new_rows = len(df_new)

    # ✅ Save merged data
    df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ Total rows after update: {len(df_combined)}")
    print(f"➕ Newly added rows: {new_rows}")
    print("📌 Latest entries:")
    print(df_combined.tail())

if __name__ == "__main__":
    fetch_air_quality_data()
