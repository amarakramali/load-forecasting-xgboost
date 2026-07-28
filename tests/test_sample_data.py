from __future__ import annotations

import pandas as pd
import pytest

from aep_load_forecasting.make_features import (
    build_features,
    load_hourly_series,
)
from aep_load_forecasting.sample_data import generate_hourly_load, main


def test_generate_hourly_load_is_deterministic_and_hourly() -> None:
    first = generate_hourly_load(days=8, seed=7)
    second = generate_hourly_load(days=8, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["Datetime", "AEP_MW"]
    assert len(first) == 8 * 24
    assert first["Datetime"].is_monotonic_increasing
    assert first["Datetime"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all()
    assert first["AEP_MW"].gt(0).all()
    assert first["AEP_MW"].nunique() > 100


def test_generate_hourly_load_changes_with_seed() -> None:
    first = generate_hourly_load(days=8, seed=1)
    second = generate_hourly_load(days=8, seed=2)

    assert not first["AEP_MW"].equals(second["AEP_MW"])
    pd.testing.assert_series_equal(first["Datetime"], second["Datetime"])


@pytest.mark.parametrize("days", [0, 7])
def test_generate_hourly_load_rejects_short_series(days: int) -> None:
    with pytest.raises(ValueError, match="At least 8 days"):
        generate_hourly_load(days=days)


@pytest.mark.parametrize("days", [8.5, True])
def test_generate_hourly_load_rejects_non_integer_days(
    days: object,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        generate_hourly_load(days=days)  # type: ignore[arg-type]


@pytest.mark.parametrize("start", ["not-a-date", "2025-01-01 00:30:00"])
def test_generate_hourly_load_rejects_invalid_start(start: str) -> None:
    with pytest.raises(ValueError, match="start timestamp|Start timestamp"):
        generate_hourly_load(days=8, start=start)


def test_command_writes_feature_compatible_csv(tmp_path) -> None:
    output = tmp_path / "nested" / "sample.csv"

    result = main(
        [
            "--output",
            str(output),
            "--days",
            "8",
            "--seed",
            "9",
        ]
    )

    assert result == 0
    assert output.is_file()
    history = load_hourly_series(output)
    features = build_features(history)
    assert len(history) == 8 * 24
    assert len(features) == 24
