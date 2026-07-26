"""Feature construction and recursive forecasting for future load."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

HOURLY_STEP = pd.Timedelta(hours=1)
MINIMUM_HISTORY_HOURS = 168
FORECAST_FEATURES = (
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_24_mean",
    "roll_168_mean",
)


class Regressor(Protocol):
    """Minimal interface required by the recursive forecaster."""

    def predict(self, features: pd.DataFrame) -> object:
        """Return one prediction for each feature row."""


def validate_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Validate that model inputs match the forecast feature contract."""

    if isinstance(columns, str):
        raise TypeError("Feature columns must be a sequence of column names.")

    normalized = tuple(columns)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Feature columns must not contain duplicates.")

    missing = sorted(set(FORECAST_FEATURES).difference(normalized))
    unexpected = sorted(set(normalized).difference(FORECAST_FEATURES))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            "Feature columns do not match the forecast contract ("
            + "; ".join(details)
            + ")."
        )
    return normalized


def _validated_history(history: pd.Series) -> pd.Series:
    if not isinstance(history, pd.Series):
        raise TypeError("History must be a pandas Series.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("History must use a DatetimeIndex.")
    if history.empty:
        raise ValueError("History contains no observations.")
    if history.index.has_duplicates:
        raise ValueError("History contains duplicate timestamps.")
    if not history.index.is_monotonic_increasing:
        raise ValueError("History must be sorted chronologically.")
    if history.isna().any():
        raise ValueError("History contains missing values.")
    if not pd.api.types.is_numeric_dtype(history):
        raise TypeError("History values must be numeric.")
    if len(history) < MINIMUM_HISTORY_HOURS:
        raise ValueError(
            "At least 168 hourly observations are required for forecasting."
        )

    steps = history.index.to_series().diff().dropna()
    if not steps.eq(HOURLY_STEP).all():
        raise ValueError("History must contain one observation per hour.")
    return history.astype(float).copy()


def _make_forecast_row(
    timestamp: pd.Timestamp,
    history: pd.Series,
) -> pd.DataFrame:
    previous_timestamp = timestamp - HOURLY_STEP
    past = history.loc[:previous_timestamp]

    row = pd.DataFrame(
        {
            "hour": [timestamp.hour],
            "dayofweek": [timestamp.dayofweek],
            "month": [timestamp.month],
            "is_weekend": [int(timestamp.dayofweek >= 5)],
            "lag_1": [history.loc[timestamp - pd.Timedelta(hours=1)]],
            "lag_24": [history.loc[timestamp - pd.Timedelta(hours=24)]],
            "lag_168": [history.loc[timestamp - pd.Timedelta(hours=168)]],
            "roll_24_mean": [past.tail(24).mean()],
            "roll_168_mean": [past.tail(168).mean()],
        },
        index=pd.DatetimeIndex([timestamp], name="Datetime"),
    )
    return row.loc[:, FORECAST_FEATURES]


def make_forecast_row(
    timestamp: str | pd.Timestamp,
    history: pd.Series,
) -> pd.DataFrame:
    """Build one leakage-safe feature row for the next hourly timestamp."""

    validated_history = _validated_history(history)
    forecast_timestamp = pd.Timestamp(timestamp)
    expected_timestamp = validated_history.index[-1] + HOURLY_STEP
    if forecast_timestamp != expected_timestamp:
        raise ValueError(
            "Forecast timestamp must be exactly one hour after history."
        )
    return _make_forecast_row(forecast_timestamp, validated_history)


def recursive_forecast(
    model: Regressor,
    history: pd.Series,
    feature_columns: Sequence[str] = FORECAST_FEATURES,
    *,
    horizon: int = 24,
) -> pd.DataFrame:
    """Forecast future hours, feeding each prediction back into history."""

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")

    columns = validate_feature_columns(feature_columns)
    working_history = _validated_history(history)
    future_index = pd.date_range(
        working_history.index[-1] + HOURLY_STEP,
        periods=horizon,
        freq="h",
        name="Datetime",
    )
    predictions: list[float] = []
    baseline_predictions: list[float] = []

    for timestamp in future_index:
        row = _make_forecast_row(timestamp, working_history)
        model_input = row.loc[:, columns]

        raw_prediction = np.asarray(
            model.predict(model_input),
            dtype=float,
        ).reshape(-1)
        if raw_prediction.size != 1:
            raise ValueError("Model must return exactly one prediction per step.")
        prediction = float(raw_prediction[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction.")

        baseline = 0.5 * float(row["lag_24"].iloc[0])
        baseline += 0.5 * float(row["lag_168"].iloc[0])

        predictions.append(prediction)
        baseline_predictions.append(baseline)
        working_history.loc[timestamp] = prediction

    return pd.DataFrame(
        {
            "forecast_xgb_MW": predictions,
            "baseline_blend_MW": baseline_predictions,
        },
        index=future_index,
    )
