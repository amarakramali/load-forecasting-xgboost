from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    FORECAST_FEATURES,
    make_forecast_row,
    recursive_forecast,
)


def hourly_history(periods: int = 200) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    return pd.Series(
        np.arange(1, periods + 1, dtype=float),
        index=index,
        name="load",
    )


class IncrementingModel:
    def __init__(self) -> None:
        self.inputs: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.inputs.append(features.copy())
        return features["lag_1"].to_numpy() + 1.0


def test_make_forecast_row_uses_only_past_values() -> None:
    history = hourly_history()
    timestamp = history.index[-1] + pd.Timedelta(hours=1)

    row = make_forecast_row(timestamp, history)

    assert tuple(row.columns) == FORECAST_FEATURES
    assert row.index.tolist() == [timestamp]
    assert row.loc[timestamp, "lag_1"] == 200.0
    assert row.loc[timestamp, "lag_24"] == 177.0
    assert row.loc[timestamp, "lag_168"] == 33.0
    assert row.loc[timestamp, "roll_24_mean"] == pytest.approx(188.5)
    assert row.loc[timestamp, "roll_168_mean"] == pytest.approx(116.5)


def test_recursive_forecast_feeds_predictions_back_as_lag_one() -> None:
    history = hourly_history()
    original_history = history.copy()
    model = IncrementingModel()

    forecast = recursive_forecast(model, history, horizon=3)

    assert forecast["forecast_xgb_MW"].tolist() == [201.0, 202.0, 203.0]
    assert forecast["baseline_blend_MW"].tolist() == [
        105.0,
        106.0,
        107.0,
    ]
    assert [frame["lag_1"].iloc[0] for frame in model.inputs] == [
        200.0,
        201.0,
        202.0,
    ]
    assert forecast.index[0] == history.index[-1] + pd.Timedelta(hours=1)
    assert forecast.index.freq == pd.offsets.Hour()
    pd.testing.assert_series_equal(history, original_history)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_recursive_forecast_rejects_invalid_horizon(horizon: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recursive_forecast(
            IncrementingModel(),
            hourly_history(),
            horizon=horizon,  # type: ignore[arg-type]
        )


def test_recursive_forecast_rejects_missing_hour() -> None:
    history = hourly_history().drop(hourly_history().index[50])

    with pytest.raises(ValueError, match="one observation per hour"):
        recursive_forecast(IncrementingModel(), history)


def test_recursive_forecast_rejects_incomplete_feature_contract() -> None:
    with pytest.raises(ValueError, match="missing"):
        recursive_forecast(
            IncrementingModel(),
            hourly_history(),
            feature_columns=FORECAST_FEATURES[:-1],
        )


class NonFiniteModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([np.nan])


def test_recursive_forecast_rejects_non_finite_prediction() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        recursive_forecast(NonFiniteModel(), hourly_history())
