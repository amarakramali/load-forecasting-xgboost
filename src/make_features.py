"""Build leakage-safe calendar and lag features for hourly load data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATETIME_COLUMN = "Datetime"
DEFAULT_TARGET = "AEP_MW"
DEFAULT_INPUT = Path("data") / "AEP_hourly.csv"
DEFAULT_OUTPUT = Path("data") / "features_aep.csv"
HOURLY_STEP = pd.Timedelta(hours=1)


def load_hourly_series(
    csv_path: str | Path,
    target: str = DEFAULT_TARGET,
) -> pd.Series:
    """Load, validate, and normalize an hourly load series.

    Duplicate timestamps are averaged, which makes daylight-saving-time
    duplicates deterministic. Missing hours are rejected because row-based lag
    features would otherwise refer to the wrong point in time.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {DATETIME_COLUMN, target}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")

    timestamps = pd.to_datetime(frame[DATETIME_COLUMN], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Input CSV contains {invalid_timestamps} invalid timestamp(s)."
        )

    values = pd.to_numeric(frame[target], errors="coerce")
    invalid_values = int(values.isna().sum())
    if invalid_values:
        raise ValueError(
            f"Column {target!r} contains {invalid_values} non-numeric or "
            "missing value(s)."
        )

    hourly = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamps, name=DATETIME_COLUMN),
        name="y",
    )
    hourly = hourly.groupby(level=0, sort=True).mean()

    expected_index = pd.date_range(
        hourly.index.min(),
        hourly.index.max(),
        freq="h",
        name=DATETIME_COLUMN,
    )
    missing_hours = expected_index.difference(hourly.index)
    if len(missing_hours):
        preview = ", ".join(str(timestamp) for timestamp in missing_hours[:3])
        suffix = "..." if len(missing_hours) > 3 else ""
        raise ValueError(
            f"Input series has {len(missing_hours)} missing hourly timestamp(s): "
            f"{preview}{suffix}"
        )

    return hourly.reindex(expected_index)


def build_features(load: pd.Series) -> pd.DataFrame:
    """Create past-only features from a complete hourly load series."""

    if not isinstance(load.index, pd.DatetimeIndex):
        raise TypeError("Load series must use a DatetimeIndex.")
    if load.empty:
        raise ValueError("Load series contains no rows.")
    if load.index.has_duplicates:
        raise ValueError("Load series contains duplicate timestamps.")
    if not load.index.is_monotonic_increasing:
        raise ValueError("Load series must be sorted chronologically.")
    if load.isna().any():
        raise ValueError("Load series contains missing values.")

    steps = load.index.to_series().diff().dropna()
    if not steps.eq(HOURLY_STEP).all():
        raise ValueError("Load series must contain one observation per hour.")
    if len(load) < 169:
        raise ValueError(
            "At least 169 hourly observations are required for lag_168."
        )

    features = pd.DataFrame(index=load.index)
    features["y"] = load.astype(float)
    features["hour"] = features.index.hour
    features["dayofweek"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)

    past_load = features["y"].shift(1)
    features["lag_1"] = past_load
    features["lag_24"] = features["y"].shift(24)
    features["lag_168"] = features["y"].shift(168)
    features["roll_24_mean"] = past_load.rolling(24).mean()
    features["roll_168_mean"] = past_load.rolling(168).mean()

    return features.dropna()


def make_feature_table(
    csv_path: str | Path,
    target: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Load an input CSV and return its validated feature table."""

    return build_features(load_hourly_series(csv_path, target=target))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe features from hourly AEP load data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT.as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Load column name (default: {DEFAULT_TARGET})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = make_feature_table(args.input, target=args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output)

    print(f"Rows: {len(features)}")
    print(f"Columns: {', '.join(features.columns)}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
