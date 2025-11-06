from datetime import timedelta
from feast import FileSource, FeatureView, Field
from feast.types import Float32
from entities import city

# ✅ Define the data source (auto-detects CSV or Parquet)
data_source = FileSource(
    path="../data/daily/clean_air_quality_data.parquet",
    timestamp_field="time",
)

# ✅ Define the Feature View
aqi_features = FeatureView(
    name="aqi_features",
    entities=[city],
    ttl=timedelta(days=1),
    schema=[
        Field(name="pm10", dtype=Float32),
        Field(name="pm2_5", dtype=Float32),
        Field(name="carbon_monoxide", dtype=Float32),
        Field(name="nitrogen_dioxide", dtype=Float32),
        Field(name="sulphur_dioxide", dtype=Float32),
        Field(name="ozone", dtype=Float32),
        Field(name="AQI", dtype=Float32),
    ],
    online=True,
    source=data_source,
)
