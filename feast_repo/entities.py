from feast import Entity, ValueType

city = Entity(
    name="city",
    join_keys=["city"],
    description="City AQI data",
    value_type=ValueType.STRING,
)
