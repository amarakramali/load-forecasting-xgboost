"""Evaluate naive load baselines and persist their artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation import HOURS_PER_DAY, trailing_window
from src.reporting import (
    EvaluationResult,
    evaluate_predictions,
    format_result,
    save_results,
)

DEFAULT_FEATURES = Path("data") / "features_aep.csv"
DEFAULT_METRICS = Path("reports") / "baseline_metrics.csv"
DEFAULT_PLOT = Path("reports") / "figures" / "baseline_evaluation.png"
DEFAULT_EVALUATION_DAYS = 30
DEFAULT_PLOT_DAYS = 7
REQUIRED_COLUMNS = ("y", "lag_24", "lag_168")
HOURLY_STEP = pd.Timedelta(hours=1)


def load_baseline_features(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the columns required for baseline evaluation."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {"Datetime", *REQUIRED_COLUMNS}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Feature CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError("Feature CSV contains no rows.")

    timestamps = pd.to_datetime(frame["Datetime"], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Feature CSV contains {invalid_timestamps} invalid timestamp(s)."
        )
    if timestamps.duplicated().any():
        raise ValueError("Feature CSV contains duplicate timestamps.")

    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_values = int(numeric.isna().sum().sum())
    if invalid_values:
        raise ValueError(
            "Feature CSV contains "
            f"{invalid_values} non-numeric or missing value(s)."
        )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature CSV contains non-finite values.")

    numeric.index = pd.DatetimeIndex(timestamps, name="Datetime")
    numeric = numeric.sort_index()
    steps = numeric.index.to_series().diff().dropna()
    if not steps.eq(HOURLY_STEP).all():
        raise ValueError(
            "Feature CSV must contain consecutive hourly timestamps."
        )
    return numeric.astype(float)


def baseline_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """Build yesterday, last-week, and blended baseline predictions."""

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(features.columns))
    if missing_columns:
        raise ValueError(
            "Evaluation data is missing required columns: "
            + ", ".join(missing_columns)
        )

    predictions = pd.DataFrame(index=features.index)
    predictions["Yesterday"] = features["lag_24"].astype(float)
    predictions["Last week"] = features["lag_168"].astype(float)
    predictions["Blend 50/50"] = (
        0.5 * predictions["Yesterday"] + 0.5 * predictions["Last week"]
    )
    return predictions


def evaluate_baselines(
    features: pd.DataFrame,
    *,
    evaluation_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[EvaluationResult]]:
    """Evaluate all baselines on an exact trailing time window."""

    test = trailing_window(features, hours=evaluation_hours)
    predictions = baseline_predictions(test)
    results = [
        evaluate_predictions(
            "Yesterday (lag_24)",
            test["y"],
            predictions["Yesterday"],
        ),
        evaluate_predictions(
            "Last week (lag_168)",
            test["y"],
            predictions["Last week"],
        ),
        evaluate_predictions(
            "Blend 50/50",
            test["y"],
            predictions["Blend 50/50"],
        ),
    ]
    return test, predictions, results


def save_baseline_plot(
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    plot_hours: int,
) -> Path:
    """Plot actual load and the blended baseline over a trailing window."""

    if not test.index.equals(predictions.index):
        raise ValueError("Test data and predictions must use the same index.")

    plot_test = trailing_window(test, hours=plot_hours)
    plot_predictions = predictions.loc[plot_test.index]
    path = Path(output_path)

    figure = Figure()
    axis = figure.subplots()
    axis.plot(plot_test.index, plot_test["y"], label="Actual")
    axis.plot(
        plot_predictions.index,
        plot_predictions["Blend 50/50"],
        label="Blend 50/50",
    )
    axis.set_title("Baseline Forecast")
    axis.set_xlabel("Time")
    axis.set_ylabel("MW")
    axis.legend()
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate naive load baselines on a trailing window."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"Feature CSV path (default: {DEFAULT_FEATURES.as_posix()})",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help=f"Metrics CSV path (default: {DEFAULT_METRICS.as_posix()})",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help=f"Plot path (default: {DEFAULT_PLOT.as_posix()})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_EVALUATION_DAYS,
        help=(
            "Number of trailing evaluation days "
            f"(default: {DEFAULT_EVALUATION_DAYS})"
        ),
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=DEFAULT_PLOT_DAYS,
        help=(
            "Number of trailing evaluation days to plot "
            f"(default: {DEFAULT_PLOT_DAYS})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = load_baseline_features(args.features)
    test, predictions, results = evaluate_baselines(
        features,
        evaluation_hours=args.days * HOURS_PER_DAY,
    )

    print(f"Baseline evaluation (last {args.days} days):")
    for result in results:
        print(format_result(result))

    saved_metrics = save_results(results, args.metrics)
    print(f"Metrics saved: {saved_metrics}")
    saved_plot = save_baseline_plot(
        test,
        predictions,
        args.plot,
        plot_hours=args.plot_days * HOURS_PER_DAY,
    )
    print(f"Plot saved: {saved_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
