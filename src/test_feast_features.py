from feast import FeatureStore
import pandas as pd

# Initialize feature store
store = FeatureStore(repo_path="feast_repo")

# Create dummy entity dataframe
entity_df = pd.DataFrame.from_dict({"city": ["Karachi"], "event_timestamp": [pd.Timestamp.now()]})

# Fetch features from Feast
features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "aqi_features:pm10",
        "aqi_features:pm2_5",
        "aqi_features:ozone",
        "aqi_features:AQI",
    ],
).to_df()

print("✅ Retrieved features from Feast:")
print(features.head())
