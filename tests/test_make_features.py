from __future__ import annotations

import pandas as pd
import pytest

from aep_load_forecasting.make_features import (
    build_features,
    load_hourly_series,
    make_feature_table,
)


def synthetic_load(periods: int = 200) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    return pd.Series(
        [1_000.0 + offset for offset in range(periods)],
        index=index,
        name="y",
    )


def write_load_csv(path, load: pd.Series) -> None:
    pd.DataFrame(
        {
            "Datetime": load.index,
            "AEP_MW": load.to_numpy(),
        }
    ).to_csv(path, index=False)


def test_build_features_uses_only_past_observations() -> None:
    load = synthetic_load()

    features = build_features(load)

    first_timestamp = load.index[168]
    first_row = features.loc[first_timestamp]
    assert first_row["y"] == load.iloc[168]
    assert first_row["lag_1"] == load.iloc[167]
    assert first_row["lag_24"] == load.iloc[144]
    assert first_row["lag_168"] == load.iloc[0]
    assert first_row["roll_24_mean"] == pytest.approx(load.iloc[144:168].mean())
    assert first_row["roll_168_mean"] == pytest.approx(load.iloc[:168].mean())


def test_load_hourly_series_sorts_and_averages_duplicate_timestamps(
    tmp_path,
) -> None:
    load = synthetic_load()
    frame = pd.DataFrame(
        {
            "Datetime": load.index,
            "AEP_MW": load.to_numpy(),
        }
    )
    duplicate = pd.DataFrame(
        {
            "Datetime": [load.index[10]],
            "AEP_MW": [load.iloc[10] + 20.0],
        }
    )
    csv_path = tmp_path / "duplicate.csv"
    pd.concat([frame.iloc[::-1], duplicate], ignore_index=True).to_csv(
        csv_path,
        index=False,
    )

    normalized = load_hourly_series(csv_path)

    assert normalized.index.is_monotonic_increasing
    assert normalized.index.is_unique
    assert normalized.loc[load.index[10]] == pytest.approx(load.iloc[10] + 10.0)


def test_load_hourly_series_rejects_missing_hours(tmp_path) -> None:
    load = synthetic_load().drop(synthetic_load().index[20])
    csv_path = tmp_path / "missing-hour.csv"
    write_load_csv(csv_path, load)

    with pytest.raises(ValueError, match="1 missing hourly timestamp"):
        load_hourly_series(csv_path)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"Datetime": ["2025-01-01"]}),
            "missing required columns: AEP_MW",
        ),
        (
            pd.DataFrame(
                {"Datetime": ["not-a-date"], "AEP_MW": [1_000.0]}
            ),
            "1 invalid timestamp",
        ),
        (
            pd.DataFrame(
                {"Datetime": ["2025-01-01"], "AEP_MW": ["not-a-number"]}
            ),
            "1 non-numeric or missing value",
        ),
    ],
)
def test_load_hourly_series_rejects_invalid_input(
    tmp_path,
    frame: pd.DataFrame,
    message: str,
) -> None:
    csv_path = tmp_path / "invalid.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=message):
        load_hourly_series(csv_path)


def test_make_feature_table_reads_csv(tmp_path) -> None:
    csv_path = tmp_path / "load.csv"
    write_load_csv(csv_path, synthetic_load())

    features = make_feature_table(csv_path)

    assert len(features) == 32
    assert list(features.columns) == [
        "y",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_24_mean",
        "roll_168_mean",
    ]
