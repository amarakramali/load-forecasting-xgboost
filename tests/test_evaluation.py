from __future__ import annotations

import pandas as pd
import pytest

from aep_load_forecasting.evaluation import (
    chronological_split,
    trailing_window,
)


def hourly_frame(periods: int = 2_000) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    return pd.DataFrame({"y": range(periods)}, index=index)


def test_chronological_split_returns_exact_disjoint_windows() -> None:
    frame = hourly_frame()

    train, validation, test = chronological_split(
        frame,
        validation_hours=30 * 24,
        test_hours=30 * 24,
    )

    assert len(train) == 560
    assert len(validation) == 720
    assert len(test) == 720
    assert train.index.intersection(validation.index).empty
    assert train.index.intersection(test.index).empty
    assert validation.index.intersection(test.index).empty
    assert validation.index.min() - train.index.max() == pd.Timedelta(hours=1)
    assert test.index.min() - validation.index.max() == pd.Timedelta(hours=1)
    assert test.index.max() == frame.index.max()


def test_chronological_split_returns_independent_frames() -> None:
    frame = hourly_frame()
    train, validation, test = chronological_split(
        frame,
        validation_hours=100,
        test_hours=100,
    )

    train.iloc[-1, 0] = -1
    validation.iloc[-1, 0] = -2
    test.iloc[-1, 0] = -3

    assert (frame["y"] >= 0).all()


def test_chronological_split_rejects_insufficient_history() -> None:
    frame = hourly_frame(periods=200)

    with pytest.raises(ValueError, match="one training observation"):
        chronological_split(
            frame,
            validation_hours=100,
            test_hours=100,
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"y": [1, 2, 3]}),
            "DatetimeIndex",
        ),
        (
            hourly_frame(periods=10).iloc[::-1],
            "sorted chronologically",
        ),
        (
            pd.concat(
                [
                    hourly_frame(periods=10),
                    hourly_frame(periods=10).iloc[[-1]],
                ]
            ).sort_index(),
            "duplicate timestamps",
        ),
    ],
)
def test_chronological_split_rejects_invalid_index(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        chronological_split(
            frame,
            validation_hours=2,
            test_hours=2,
        )


def test_trailing_window_returns_exact_number_of_rows() -> None:
    frame = hourly_frame(periods=1_000)

    test = trailing_window(frame, hours=30 * 24)

    assert len(test) == 720
    assert test.index.max() == frame.index.max()
    assert test.index.min() == frame.index[-720]
