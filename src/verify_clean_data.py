import pandas as pd
import matplotlib.pyplot as plt
import os

clean_path = "data/daily/clean_air_quality_data.csv"

if not os.path.exists(clean_path):
    print("❌ Cleaned file not found. Run clean_data.py first!")
    exit()

# Step 1: Load the cleaned file
df = pd.read_csv(clean_path)
print(f"✅ Loaded cleaned data successfully! Shape: {df.shape}")

# Step 2: Check for missing values
print("\n🔍 Missing values per column:")
print(df.replace([" ", "", "NA", "N/A", "None"], pd.NA).isna().sum())

# Step 3: Check duplicates
if "time" in df.columns:
    duplicates = df["time"].duplicated().sum()
    print(f"\n🔁 Duplicate timestamps: {duplicates}")
else:
    print("⚠️ No 'time' column found to check duplicates.")

# Step 4: Range check for pollutants
cols = ["pm10", "pm2_5", "sulphur_dioxide", "nitrogen_dioxide", "carbon_monoxide", "ozone"]
print("\n📊 Pollutant value ranges:")
for c in cols:
    if c in df.columns:
        print(f"{c}: Min={df[c].min()}, Max={df[c].max()}")
    else:
        print(f"⚠️ Column {c} missing in data.")

# Step 5: Basic statistics
print("\n📈 Summary statistics:")
print(df.describe())
print("\n🎉 Verification complete! Data looks clean if no warnings above.")
