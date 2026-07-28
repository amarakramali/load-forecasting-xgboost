"""Validation helpers for forecast CSV files shown in the demo."""

from __future__ import annotations

from pathlib import Path
from typing import IO, TypeAlias

import numpy as np
import pandas as pd

CsvSource: TypeAlias = str | Path | IO[str] | IO[bytes]

DATETIME_COLUMN = "Datetime"
FORECAST_COLUMN = "forecast_xgb_MW"
BASELINE_COLUMN = "baseline_blend_MW"
LOWER_COLUMN = "forecast_xgb_lower_MW"
UPPER_COLUMN = "forecast_xgb_upper_MW"
PLOT_COLUMNS = (
    FORECAST_COLUMN,
    LOWER_COLUMN,
    UPPER_COLUMN,
    BASELINE_COLUMN,
)
HOURLY_STEP = pd.Timedelta(hours=1)


class ForecastDataError(ValueError):
    """Raised when a forecast CSV cannot be displayed safely."""


def _read_csv(source: CsvSource) -> pd.DataFrame:
    try:
        return pd.read_csv(source)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise ForecastDataError(
            f"Could not read forecast CSV: {error}"
        ) from error


def _timestamp_column(frame: pd.DataFrame) -> str:
    if DATETIME_COLUMN in frame.columns:
        return DATETIME_COLUMN

    first_column = str(frame.columns[0])
    if first_column.startswith("Unnamed:"):
        return first_column

    raise ForecastDataError(
        "Forecast CSV must contain a 'Datetime' column."
    )


def load_forecast_csv(source: CsvSource) -> pd.DataFrame:
    """Read and validate a forecast CSV for display in the Streamlit app."""

    frame = _read_csv(source)
    if frame.empty:
        raise ForecastDataError("Forecast CSV contains no rows.")

    timestamp_column = _timestamp_column(frame)
    if FORECAST_COLUMN not in frame.columns:
        raise ForecastDataError(
            f"Forecast CSV is missing required column {FORECAST_COLUMN!r}."
        )
    interval_columns = {LOWER_COLUMN, UPPER_COLUMN}
    present_interval_columns = interval_columns.intersection(frame.columns)
    if present_interval_columns and present_interval_columns != interval_columns:
        raise ForecastDataError(
            "Forecast CSV must contain both prediction-interval columns."
        )

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise ForecastDataError(
            "Forecast CSV contains "
            f"{invalid_timestamps} invalid timestamp(s)."
        )
    if timestamps.duplicated().any():
        raise ForecastDataError(
            "Forecast CSV contains duplicate timestamps."
        )

    numeric_columns = [
        column for column in PLOT_COLUMNS if column in frame.columns
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_values = int(values.isna().sum())
        if invalid_values:
            raise ForecastDataError(
                f"Column {column!r} contains {invalid_values} "
                "non-numeric or missing value(s)."
            )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ForecastDataError(
                f"Column {column!r} contains non-finite values."
            )
        frame[column] = values.astype(float)

    if present_interval_columns:
        outside_interval = (
            (frame[LOWER_COLUMN] > frame[FORECAST_COLUMN])
            | (frame[FORECAST_COLUMN] > frame[UPPER_COLUMN])
        )
        if outside_interval.any():
            raise ForecastDataError(
                "Prediction intervals must contain the point forecast."
            )

    frame = frame.drop(columns=timestamp_column)
    frame.index = pd.DatetimeIndex(timestamps, name=DATETIME_COLUMN)
    frame = frame.sort_index()

    steps = frame.index.to_series().diff().dropna()
    if not steps.eq(HOURLY_STEP).all():
        raise ForecastDataError(
            "Forecast timestamps must be consecutive hourly values."
        )
    return frame


def forecast_plot_columns(frame: pd.DataFrame) -> list[str]:
    """Return forecast columns in a stable chart order."""

    return [column for column in PLOT_COLUMNS if column in frame.columns]
