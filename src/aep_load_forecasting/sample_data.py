"""Generate deterministic synthetic hourly load data for a local demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path("data") / "sample_aep_hourly.csv"
DEFAULT_START = "2025-01-01 00:00:00"
DEFAULT_DAYS = 90
DEFAULT_SEED = 42
HOURS_PER_DAY = 24
MINIMUM_DAYS = 8


def generate_hourly_load(
    *,
    days: int = DEFAULT_DAYS,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return an AEP-shaped synthetic series with realistic seasonality.

    The signal combines a daily demand cycle, weekday/weekend behavior,
    a slower monthly cycle, a small trend, and deterministic random noise.
    It is intended for exercising the pipeline, not for model benchmarking.
    """

    if isinstance(days, bool) or not isinstance(days, int):
        raise TypeError("Days must be an integer.")
    if days < MINIMUM_DAYS:
        raise ValueError(
            f"At least {MINIMUM_DAYS} days are required to build lag features."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    try:
        start_timestamp = pd.Timestamp(start)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid start timestamp: {start!r}") from error
    if pd.isna(start_timestamp):
        raise ValueError(f"Invalid start timestamp: {start!r}")
    if start_timestamp != start_timestamp.floor("h"):
        raise ValueError("Start timestamp must be aligned to a full hour.")

    periods = days * HOURS_PER_DAY
    index = pd.date_range(
        start_timestamp,
        periods=periods,
        freq="h",
        name="Datetime",
    )
    elapsed_hours = np.arange(periods, dtype=float)
    hour_of_day = index.hour.to_numpy(dtype=float)
    day_of_week = index.dayofweek.to_numpy()

    daily_cycle = 2_200.0 * np.sin(
        2.0 * np.pi * (hour_of_day - 7.0) / HOURS_PER_DAY
    )
    second_daily_peak = 700.0 * np.sin(
        4.0 * np.pi * (hour_of_day - 15.0) / HOURS_PER_DAY
    )
    weekend_effect = np.where(day_of_week >= 5, -900.0, 0.0)
    monthly_cycle = 450.0 * np.sin(
        2.0 * np.pi * elapsed_hours / (30.0 * HOURS_PER_DAY)
    )
    trend = 0.08 * elapsed_hours
    noise = np.random.default_rng(seed).normal(
        loc=0.0,
        scale=180.0,
        size=periods,
    )

    load = (
        14_500.0
        + daily_cycle
        + second_daily_peak
        + weekend_effect
        + monthly_cycle
        + trend
        + noise
    )
    return pd.DataFrame(
        {
            "Datetime": index,
            "AEP_MW": np.round(load, 3),
        }
    )


def save_sample_data(
    data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist generated data and create its parent directory."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic AEP-shaped hourly demo data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days to generate (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"First hourly timestamp (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = generate_hourly_load(
        days=args.days,
        start=args.start,
        seed=args.seed,
    )
    saved_path = save_sample_data(data, args.output)
    print(f"Rows: {len(data)}")
    print(f"Range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"Saved: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
