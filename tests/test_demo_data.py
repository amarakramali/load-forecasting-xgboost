from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from src.demo_data import (
    BASELINE_COLUMN,
    FORECAST_COLUMN,
    ForecastDataError,
    forecast_plot_columns,
    load_forecast_csv,
)


def csv_source(contents: str) -> StringIO:
    return StringIO(contents.strip())


def test_load_forecast_csv_accepts_generated_index_format() -> None:
    source = csv_source(
        """
,forecast_xgb_MW,baseline_blend_MW
2025-01-01 00:00:00,120.5,118
2025-01-01 01:00:00,121.5,119
"""
    )

    forecast = load_forecast_csv(source)

    assert forecast.index.name == "Datetime"
    assert forecast.index.tolist() == list(
        pd.date_range("2025-01-01", periods=2, freq="h")
    )
    assert forecast[FORECAST_COLUMN].tolist() == [120.5, 121.5]
    assert forecast[BASELINE_COLUMN].tolist() == [118.0, 119.0]
    assert forecast_plot_columns(forecast) == [
        FORECAST_COLUMN,
        BASELINE_COLUMN,
    ]


def test_load_forecast_csv_sorts_named_timestamps() -> None:
    source = csv_source(
        """
Datetime,forecast_xgb_MW
2025-01-01 01:00:00,121
2025-01-01 00:00:00,120
"""
    )

    forecast = load_forecast_csv(source)

    assert forecast[FORECAST_COLUMN].tolist() == [120.0, 121.0]
    assert forecast_plot_columns(forecast) == [FORECAST_COLUMN]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            """
Datetime,baseline_blend_MW
2025-01-01 00:00:00,118
""",
            "missing required column",
        ),
        (
            """
Datetime,forecast_xgb_MW
not-a-date,120
""",
            "invalid timestamp",
        ),
        (
            """
Datetime,forecast_xgb_MW
2025-01-01 00:00:00,not-a-number
""",
            "non-numeric",
        ),
        (
            """
Datetime,forecast_xgb_MW
2025-01-01 00:00:00,120
2025-01-01 00:00:00,121
""",
            "duplicate timestamps",
        ),
        (
            """
Datetime,forecast_xgb_MW
2025-01-01 00:00:00,120
2025-01-01 02:00:00,121
""",
            "consecutive hourly",
        ),
        (
            """
timestamp,forecast_xgb_MW
2025-01-01 00:00:00,120
""",
            "'Datetime' column",
        ),
    ],
)
def test_load_forecast_csv_rejects_invalid_data(
    contents: str,
    message: str,
) -> None:
    with pytest.raises(ForecastDataError, match=message):
        load_forecast_csv(csv_source(contents))


def test_load_forecast_csv_rejects_header_only_file() -> None:
    with pytest.raises(ForecastDataError, match="no rows"):
        load_forecast_csv(
            csv_source("Datetime,forecast_xgb_MW"),
        )
