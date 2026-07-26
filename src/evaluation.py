"""Utilities for leakage-safe, chronological forecast evaluation."""

from __future__ import annotations

import pandas as pd

HOURS_PER_DAY = 24


def validate_time_index(frame: pd.DataFrame) -> None:
    """Validate the index assumptions required by chronological splits."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Evaluation data must use a DatetimeIndex.")
    if frame.empty:
        raise ValueError("Evaluation data contains no rows.")
    if frame.index.has_duplicates:
        raise ValueError("Evaluation data contains duplicate timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Evaluation data must be sorted chronologically.")


def chronological_split(
    frame: pd.DataFrame,
    *,
    validation_hours: int,
    test_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint train, validation, and test partitions.

    The most recent ``test_hours`` observations form the test set. The
    preceding ``validation_hours`` observations form the validation set, and
    every earlier observation is used for training.
    """

    validate_time_index(frame)
    if validation_hours <= 0:
        raise ValueError("validation_hours must be greater than zero.")
    if test_hours <= 0:
        raise ValueError("test_hours must be greater than zero.")

    held_out_hours = validation_hours + test_hours
    if len(frame) <= held_out_hours:
        raise ValueError(
            "Evaluation data needs at least one training observation in "
            f"addition to {held_out_hours} held-out observations."
        )

    train_end = len(frame) - held_out_hours
    validation_end = len(frame) - test_hours

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    return train, validation, test


def trailing_window(frame: pd.DataFrame, *, hours: int) -> pd.DataFrame:
    """Return an exact-size trailing evaluation window."""

    validate_time_index(frame)
    if hours <= 0:
        raise ValueError("hours must be greater than zero.")
    if len(frame) < hours:
        raise ValueError(
            f"Evaluation data has {len(frame)} rows; {hours} are required."
        )
    return frame.iloc[-hours:].copy()
