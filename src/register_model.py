import os
import joblib
import hopsworks

def register_models():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "open_meteo")

    if not api_key:
        raise RuntimeError("🚨 Missing HOPSWORKS_API_KEY. Please set it in your .env file or GitHub Secrets.")

    print("🔐 Logging in to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    project._connection._model_serving_api = None  # ✅ disable serving check that hangs

    mr = project.get_model_registry()

    models_dir = "models"
    metrics = {
        "h1": {"mae": 9.51, "rmse": 13.15},
        "h2": {"mae": 15.00, "rmse": 20.45},
        "h3": {"mae": 16.84, "rmse": 22.42},
    }

    for horizon in [1, 2, 3]:
        model_file = os.path.join(models_dir, f"model_h{horizon}.joblib")

        if not os.path.exists(model_file):
            print(f"⚠️ Model file not found: {model_file}")
            continue

        print(f"📦 Registering model_h{horizon} ...")

        model = joblib.load(model_file)

        model_meta = mr.python.create_model(
            name=f"aqi_model_h{horizon}",
            metrics=metrics[f"h{horizon}"],
            model_schema={"framework": "sklearn", "version": "1.0"},
            description=f"RandomForest AQI forecast for +{horizon} day(s)"
        )

        model_resource = mr.python.register_model(model_meta, model)
        print(f"✅ Registered model_h{horizon} → version {model_meta.version}")

    print("🌟 All models registered successfully!")

if __name__ == "__main__":
    register_models()

