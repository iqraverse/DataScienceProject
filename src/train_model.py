import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

HORIZONS = [1, 2, 3]  # Predict next 1,2,3 days

def load_data():
    df = pd.read_csv("data/daily/training_features.csv", parse_dates=["time"])
    df = df.sort_values("time")
    return df

def split_train_test(df, horizon):
    df = df.copy()
    df[f"target_{horizon}"] = df["AQI"].shift(-horizon)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c not in ["time", "AQI", f"target_{horizon}"]]
    X = df[feature_cols].values
    y = df[f"target_{horizon}"].values
    return X, y, feature_cols

def train_model():
    df = load_data()
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for h in HORIZONS:
        X, y, feature_cols = split_train_test(df, h)

        split = int(len(X)*0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        # rmse = mean_squared_error(y_test, pred, squared=False)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        print(f"\n📌 Model H+{h}d:")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        joblib.dump(model, f"{models_dir}/model_h{h}.joblib")
    print("\n✅ All 3 models trained & saved!")

if __name__ == "__main__":
    train_model()