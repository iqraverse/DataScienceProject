import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="centered")

st.title("🌤️ 3-Day AQI Forecast Dashboard")
st.caption("Automatically updated via GitHub Actions")

# Load predictions
df = pd.read_csv("data/daily/predictions.csv")

st.subheader("📈 AQI Forecast (Line Chart)")
line_chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x="date:T",
        y="predicted_AQI:Q",   # <-- fixed here
        tooltip=["date", "predicted_AQI"]  # <-- and here
    )
    .properties(width=700, height=350)
)
st.altair_chart(line_chart, use_container_width=True)

st.subheader("📋 Detailed AQI Predictions")
st.dataframe(df.style.format({"predicted_AQI": "{:.2f}"}))  # <-- and here
