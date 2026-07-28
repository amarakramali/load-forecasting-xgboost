from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aep_load_forecasting.forecasting import (
    FORECAST_FEATURES,
    add_prediction_intervals,
    calibrate_recursive_intervals,
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


class PersistenceModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["lag_1"].to_numpy()


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


def test_recursive_interval_calibration_is_lead_specific() -> None:
    history = hourly_history()
    calibration_index = history.index[-6:]

    widths = calibrate_recursive_intervals(
        PersistenceModel(),
        history,
        calibration_index,
        horizon=3,
        coverage=0.5,
    )

    assert widths.index.tolist() == [1, 2, 3]
    assert widths.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.1, True])
def test_recursive_interval_calibration_rejects_invalid_coverage(
    coverage: object,
) -> None:
    history = hourly_history()

    with pytest.raises(ValueError, match="between zero and one"):
        calibrate_recursive_intervals(
            PersistenceModel(),
            history,
            history.index[-3:],
            horizon=3,
            coverage=coverage,  # type: ignore[arg-type]
        )


def test_add_prediction_intervals_clips_negative_load_bound() -> None:
    forecast = pd.DataFrame(
        {"forecast_xgb_MW": [2.0, 5.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="h"),
    )

    result = add_prediction_intervals(forecast, [3.0, 1.0])

    assert result["forecast_xgb_lower_MW"].tolist() == [0.0, 4.0]
    assert result["forecast_xgb_upper_MW"].tolist() == [5.0, 6.0]
    assert "forecast_xgb_lower_MW" not in forecast.columns


@pytest.mark.parametrize("widths", [[1.0], [1.0, -1.0], [1.0, np.nan]])
def test_add_prediction_intervals_rejects_invalid_widths(
    widths: list[float],
) -> None:
    forecast = pd.DataFrame({"forecast_xgb_MW": [2.0, 5.0]})

    with pytest.raises(ValueError, match="interval width|finite"):
        add_prediction_intervals(forecast, widths)
