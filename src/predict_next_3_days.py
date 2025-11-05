import pandas as pd
import joblib
from datetime import timedelta

def predict_next_days(features_file="data/daily/training_features.csv"):
    df = pd.read_csv(features_file, parse_dates=["time"])
    df = df.sort_values("time")

    latest = df.iloc[-1:]  # last available day data
    feature_cols = [c for c in df.columns if c not in ["time", "AQI"]]
    X_latest = latest[feature_cols].values

    results = []
    for h in [1, 2, 3]:
        model_path = f"models/model_h{h}.joblib"
        model = joblib.load(model_path)
        pred = model.predict(X_latest)[0]

        pred_date = latest["time"].iloc[0] + timedelta(days=h)
        results.append({
            "Predicted Date": pred_date.date(),
            "Days Ahead": h,
            "Predicted AQI": round(pred, 2)
        })

    preds_df = pd.DataFrame(results)
    print("\n📌 Next 3-Day AQI Forecast:")
    print(preds_df)

    preds_df.to_csv("data/daily/predictions.csv", index=False)
    print("\n✅ Prediction saved → data/daily/predictions.csv")

if __name__ == "__main__":
    predict_next_days()
