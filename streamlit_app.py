from __future__ import annotations

from pathlib import Path

import streamlit as st

from aep_load_forecasting.demo_data import (
    FORECAST_COLUMN,
    ForecastDataError,
    forecast_plot_columns,
    load_forecast_csv,
)

DEFAULT_FORECAST_PATH = Path("assets") / "forecast_next24h.csv"


def main() -> None:
    st.set_page_config(page_title="Load Forecast Demo", layout="wide")
    st.title("⚡ Load Forecast (AEP) — Demo")

    st.sidebar.header("Datenquelle")
    uploaded = st.sidebar.file_uploader(
        "Optional: eigene Forecast-CSV hochladen",
        type=["csv"],
    )

    if uploaded is not None:
        source = uploaded
        source_message = "Upload geladen und validiert."
    elif DEFAULT_FORECAST_PATH.exists():
        source = DEFAULT_FORECAST_PATH
        source_message = (
            "Nutze validierte Standarddatei: "
            f"{DEFAULT_FORECAST_PATH.as_posix()}"
        )
    else:
        st.error(
            "Keine Daten gefunden. Bitte CSV hochladen oder "
            "assets/forecast_next24h.csv bereitstellen."
        )
        st.stop()
        return

    try:
        forecast = load_forecast_csv(source)
    except ForecastDataError as error:
        st.error(f"Forecast-Datei ist ungültig: {error}")
        st.stop()
        return

    st.sidebar.success(source_message)
    st.sidebar.caption(
        f"{len(forecast)} Stunden · "
        f"{forecast.index.min()} bis {forecast.index.max()}"
    )

    st.subheader("Vorschau")
    st.dataframe(forecast.head(10), width="stretch")

    st.subheader("Forecast-Chart")
    st.line_chart(
        forecast[forecast_plot_columns(forecast)],
        width="stretch",
    )

    values = forecast[FORECAST_COLUMN]
    st.subheader("Kurz-Kennzahlen")
    maximum, minimum, average = st.columns(3)
    maximum.metric("Max Forecast (MW)", f"{values.max():.0f}")
    minimum.metric("Min Forecast (MW)", f"{values.min():.0f}")
    average.metric("Ø Forecast (MW)", f"{values.mean():.0f}")


if __name__ == "__main__":
    main()
