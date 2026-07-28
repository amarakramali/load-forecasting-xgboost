"""Feature construction and recursive forecasting for future load."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
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


def calibrate_recursive_intervals(
    model: Regressor,
    history: pd.Series,
    calibration_index: pd.DatetimeIndex,
    feature_columns: Sequence[str] = FORECAST_FEATURES,
    *,
    horizon: int = 24,
    coverage: float = 0.9,
) -> pd.Series:
    """Calibrate lead-specific interval widths on recursive forecasts.

    The model must be trained only on observations before ``calibration_index``.
    Each non-overlapping calibration block is forecast recursively from the
    history available at its origin. Absolute errors are converted to the
    finite-sample split-conformal order statistic independently for every lead.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    if isinstance(coverage, bool) or not 0.0 < coverage < 1.0:
        raise ValueError("Interval coverage must be between zero and one.")
    if not isinstance(calibration_index, pd.DatetimeIndex):
        raise TypeError("Calibration timestamps must use a DatetimeIndex.")
    if calibration_index.has_duplicates:
        raise ValueError("Calibration timestamps contain duplicates.")
    if not calibration_index.is_monotonic_increasing:
        raise ValueError("Calibration timestamps must be sorted.")
    if len(calibration_index) < horizon:
        raise ValueError(
            "Calibration data must contain at least one complete forecast "
            f"horizon ({horizon} rows)."
        )

    steps = calibration_index.to_series().diff().dropna()
    if not steps.eq(HOURLY_STEP).all():
        raise ValueError("Calibration timestamps must be consecutive hours.")

    columns = validate_feature_columns(feature_columns)
    validated_history = _validated_history(history)
    missing_timestamps = calibration_index.difference(validated_history.index)
    if not missing_timestamps.empty:
        raise ValueError("Calibration timestamps are missing from history.")

    complete_blocks = len(calibration_index) // horizon
    errors_by_lead: list[list[float]] = [[] for _ in range(horizon)]
    for block_number in range(complete_blocks):
        start = block_number * horizon
        block = calibration_index[start : start + horizon]
        past = validated_history.loc[: block[0] - HOURLY_STEP]
        forecast = recursive_forecast(
            model,
            past,
            columns,
            horizon=horizon,
        )
        if not forecast.index.equals(block):
            raise ValueError(
                "Calibration timestamps must begin immediately after the "
                "available history."
            )

        actual = validated_history.loc[block].to_numpy(dtype=float)
        predicted = forecast["forecast_xgb_MW"].to_numpy(dtype=float)
        for lead, error in enumerate(np.abs(actual - predicted)):
            errors_by_lead[lead].append(float(error))

    half_widths = []
    for errors in errors_by_lead:
        rank = min(len(errors), ceil((len(errors) + 1) * coverage))
        half_widths.append(float(np.partition(errors, rank - 1)[rank - 1]))

    return pd.Series(
        half_widths,
        index=pd.RangeIndex(1, horizon + 1, name="horizon_hour"),
        name="interval_half_width_MW",
    )


def add_prediction_intervals(
    forecast: pd.DataFrame,
    half_widths: Sequence[float],
) -> pd.DataFrame:
    """Add physically bounded symmetric intervals to a point forecast."""

    if "forecast_xgb_MW" not in forecast.columns:
        raise ValueError("Forecast is missing the forecast_xgb_MW column.")
    widths = np.asarray(half_widths, dtype=float).reshape(-1)
    if widths.size != len(forecast):
        raise ValueError("One interval width is required per forecast row.")
    if not np.isfinite(widths).all() or (widths < 0).any():
        raise ValueError("Interval widths must be finite and non-negative.")

    result = forecast.copy()
    point = result["forecast_xgb_MW"].to_numpy(dtype=float)
    result["forecast_xgb_lower_MW"] = np.maximum(0.0, point - widths)
    result["forecast_xgb_upper_MW"] = point + widths
    return result
