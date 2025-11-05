import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(line_buffering=True)

RAW_PATH = "data/raw/air_quality_data.csv"
DAILY_OUT = "data/daily/clean_air_quality_data.csv"

REQ_COLS = ["time","pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone"]

# ---------- Load ----------
if not os.path.exists(RAW_PATH):
    xl = "data/raw/air_quality_data.xlsx"
    if os.path.exists(xl):
        print("⚠️ CSV not found, reading Excel...")
        df = pd.read_excel(xl)
    else:
        print("❌ No data file found"); raise SystemExit(1)
else:
    print("✅ Loading raw CSV...")
    df = pd.read_csv(RAW_PATH)

print(f"Shape raw: {df.shape}")

# ---------- Basic schema ----------
if "time" not in df.columns:
    raise ValueError("time column missing from raw data")

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# de-duplicate strictly by timestamp
dups = df.duplicated(subset=["time"]).sum()
if dups:
    print(f"🧹 Removing {dups} duplicate timestamps")
    df = df.drop_duplicates(subset=["time"], keep="first")

# ---------- Standardize units (Open-Meteo = µg/m³ for these) ----------
pollutants = ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone"]
for col in pollutants:
    if col not in df.columns:
        print(f"⚠️ Missing column in API response: {col}")

# Fix CO if it was mistakenly saved in mg/m³ earlier
if "carbon_monoxide" in df.columns and df["carbon_monoxide"].notna().any():
    if df["carbon_monoxide"].max() < 100:  # likely mg/m³
        print("🔁 Converting CO from mg/m³ → µg/m³")
        df["carbon_monoxide"] = df["carbon_monoxide"] * 1000

# ---------- Clean numeric columns ----------
for col in pollutants:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Hard physical bounds (kept wide; all µg/m³)
bounds = {
    "pm2_5": (0, 1000),
    "pm10": (0, 1500),
    "carbon_monoxide": (0, 10000),  # µg/m³ wide cap
    "nitrogen_dioxide": (0, 2000),
    "sulphur_dioxide": (0, 2000),
    "ozone": (0, 1000),
}
for col, (lo, hi) in bounds.items():
    if col in df.columns:
        before = len(df)
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        after = len(df)
        # (rows not dropped here; values NaN → will be interpolated)

# ---------- Interpolate time-wise ----------
df = df.set_index("time").sort_index()

# ensure hourly frequency; if API has gaps, create them then interpolate
df = df.asfreq("H")

for col in pollutants:
    if col in df.columns:
        # linear time interpolation for gaps up to 6 hours; longer gaps leave NaN
        df[col] = df[col].interpolate(method="time", limit=6)
        # final small remaining NaNs: forward/back fill
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

# ---------- OPTIONAL: light winsorization to reduce extreme spikes ----------
for col in ["pm2_5","pm10","ozone","nitrogen_dioxide","sulphur_dioxide","carbon_monoxide"]:
    if col in df.columns:
        q_low, q_hi = df[col].quantile([0.001, 0.999])
        df[col] = df[col].clip(lower=q_low, upper=q_hi)

# ---------- Daily aggregation (24h mean) ----------
daily = df.resample("D").mean(numeric_only=True)

# keep also counts to verify coverage (optional)
coverage = df[pollutants].resample("D").count()
daily["obs_hours"] = coverage.max(axis=1)  # how many hours available that day
# filter days with too few hours, e.g., < 18 hours of data
daily = daily[daily["obs_hours"] >= 18].drop(columns=["obs_hours"])

# ---------- Calculate AQI from PM2.5 ----------
def calculate_aqi_pm25(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for low, high, aqi_low, aqi_high in breakpoints:
        if low <= pm25 <= high:
            return ((aqi_high - aqi_low) / (high - low)) * (pm25 - low) + aqi_low
    return None

daily['AQI'] = daily['pm2_5'].apply(calculate_aqi_pm25)

# ---------- Save ----------
os.makedirs("data/daily", exist_ok=True)
daily.reset_index().to_csv(DAILY_OUT, index=False)
print(f"✅ Clean daily file: {os.path.abspath(DAILY_OUT)}  | rows={len(daily)}")
print(daily.tail(3))
