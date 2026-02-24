import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Load Forecast Demo", layout="wide")
st.title("⚡ Load Forecast (AEP) — Demo")

default_path = Path("assets") / "forecast_next24h.csv"

st.sidebar.header("Datenquelle")
uploaded = st.sidebar.file_uploader("Optional: eigene forecast_next24h.csv hochladen", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
    st.sidebar.success("Upload geladen.")
elif default_path.exists():
    df = pd.read_csv(default_path, index_col=0, parse_dates=True)
    st.sidebar.info(f"Nutze Standarddatei: {default_path.as_posix()}")
else:
    st.error("Keine Daten gefunden. Bitte CSV hochladen oder assets/forecast_next24h.csv bereitstellen.")
    st.stop()

st.subheader("Vorschau")
st.dataframe(df.head(10), use_container_width=True)

# Erwartete Spalten aus Ihrem Script:
# forecast_xgb_MW, baseline_blend_MW
cols = [c for c in ["forecast_xgb_MW", "baseline_blend_MW"] if c in df.columns]

st.subheader("Forecast-Chart")
if cols:
    st.line_chart(df[cols], use_container_width=True)
else:
    st.line_chart(df, use_container_width=True)

st.subheader("Kurz-Kennzahlen")
c1, c2, c3 = st.columns(3)
if "forecast_xgb_MW" in df.columns:
    c1.metric("Max Forecast (MW)", f"{df['forecast_xgb_MW'].max():.0f}")
    c2.metric("Min Forecast (MW)", f"{df['forecast_xgb_MW'].min():.0f}")
    c3.metric("Ø Forecast (MW)", f"{df['forecast_xgb_MW'].mean():.0f}")
else:
    c1.metric("Zeilen", f"{len(df)}")
    c2.metric("Spalten", f"{df.shape[1]}")
    c3.metric("Start", f"{df.index.min()}")