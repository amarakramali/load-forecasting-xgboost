"""Evaluate XGBoost against the blended naive load baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from xgboost import XGBRegressor

from aep_load_forecasting.cli import add_version_argument
from aep_load_forecasting.evaluation import (
    HOURS_PER_DAY,
    chronological_split,
    trailing_window,
)
from aep_load_forecasting.forecasting import (
    FORECAST_FEATURES,
    validate_feature_columns,
)
from aep_load_forecasting.reporting import (
    EvaluationResult,
    evaluate_predictions,
    format_result,
    mae_improvement_percent,
    save_results,
)

DEFAULT_FEATURES = Path("data") / "features_aep.csv"
DEFAULT_METRICS = Path("reports") / "xgb_evaluation_metrics.csv"
DEFAULT_PLOT = Path("reports") / "figures" / "xgb_evaluation.png"
DEFAULT_EVALUATION_DAYS = 30
DEFAULT_PLOT_DAYS = 7
DEFAULT_ESTIMATORS = 800
HOURLY_STEP = pd.Timedelta(hours=1)


class EvaluationRegressor(Protocol):
    """Minimal estimator interface required by the evaluation workflow."""

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        eval_set: list[tuple[pd.DataFrame, pd.Series]],
        verbose: bool,
    ) -> object:
        """Fit the estimator with one validation set."""

    def predict(self, features: pd.DataFrame) -> object:
        """Return one prediction per feature row."""


@dataclass(frozen=True)
class XgbEvaluation:
    """Evaluation partitions, predictions, and calculated metrics."""

    test: pd.DataFrame
    baseline: pd.Series
    predictions: pd.Series
    results: tuple[EvaluationResult, EvaluationResult]
    improvement_percent: float | None


def load_evaluation_features(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the complete feature table used for evaluation."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"Datetime", "y", *FORECAST_FEATURES}
    missing_columns = sorted(required_columns.difference(frame.columns))
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

    ordered_columns = ("y", *FORECAST_FEATURES)
    numeric = frame.loc[:, ordered_columns].apply(
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


def build_xgb_model(
    *,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> XGBRegressor:
    """Create the deterministic XGBoost model used for evaluation."""

    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )


def evaluate_xgboost(
    features: pd.DataFrame,
    *,
    evaluation_hours: int,
    model: EvaluationRegressor | None = None,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> XgbEvaluation:
    """Evaluate one estimator on chronological validation and test windows."""

    if "y" not in features.columns:
        raise ValueError("Evaluation data is missing the target column: y")
    feature_columns = validate_feature_columns(
        [column for column in features.columns if column != "y"]
    )
    train, validation, test = chronological_split(
        features,
        validation_hours=evaluation_hours,
        test_hours=evaluation_hours,
    )
    estimator = (
        model
        if model is not None
        else build_xgb_model(n_estimators=n_estimators)
    )
    estimator.fit(
        train.loc[:, feature_columns],
        train["y"],
        eval_set=[(validation.loc[:, feature_columns], validation["y"])],
        verbose=False,
    )

    raw_predictions = np.asarray(
        estimator.predict(test.loc[:, feature_columns]),
        dtype=float,
    ).reshape(-1)
    if raw_predictions.size != len(test):
        raise ValueError("Model must return one prediction per test row.")
    predictions = pd.Series(
        raw_predictions,
        index=test.index,
        name="XGBoost",
    )
    baseline = (
        0.5 * test["lag_24"] + 0.5 * test["lag_168"]
    ).rename("Baseline Blend")

    baseline_result = evaluate_predictions(
        "Baseline Blend",
        test["y"],
        baseline,
    )
    xgb_result = evaluate_predictions("XGBoost", test["y"], predictions)
    improvement = None
    if baseline_result.mae_mw > 0:
        improvement = mae_improvement_percent(baseline_result, xgb_result)

    return XgbEvaluation(
        test=test,
        baseline=baseline,
        predictions=predictions,
        results=(baseline_result, xgb_result),
        improvement_percent=improvement,
    )


def save_evaluation_plot(
    evaluation: XgbEvaluation,
    output_path: str | Path,
    *,
    plot_hours: int,
) -> Path:
    """Save actual, baseline, and XGBoost values over a trailing window."""

    plot_test = trailing_window(evaluation.test, hours=plot_hours)
    plot_index = plot_test.index
    path = Path(output_path)

    figure = Figure()
    axis = figure.subplots()
    axis.plot(plot_index, plot_test["y"], label="Actual")
    axis.plot(
        plot_index,
        evaluation.baseline.loc[plot_index],
        label="Baseline Blend",
    )
    axis.plot(
        plot_index,
        evaluation.predictions.loc[plot_index],
        label="XGBoost",
    )
    axis.set_title("XGBoost Evaluation")
    axis.set_xlabel("Time")
    axis.set_ylabel("MW")
    axis.legend()
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost against a blended naive load baseline."
    )
    add_version_argument(parser)
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
            "Days in each validation and test window "
            f"(default: {DEFAULT_EVALUATION_DAYS})"
        ),
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=DEFAULT_PLOT_DAYS,
        help=(
            "Number of trailing test days to plot "
            f"(default: {DEFAULT_PLOT_DAYS})"
        ),
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=DEFAULT_ESTIMATORS,
        help=f"Number of boosting rounds (default: {DEFAULT_ESTIMATORS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = load_evaluation_features(args.features)
    evaluation = evaluate_xgboost(
        features,
        evaluation_hours=args.days * HOURS_PER_DAY,
        n_estimators=args.estimators,
    )

    print(f"Comparison (test: last {args.days} days)")
    for result in evaluation.results:
        print(format_result(result))
    saved_metrics = save_results(evaluation.results, args.metrics)
    print(f"Metrics saved: {saved_metrics}")

    if evaluation.improvement_percent is None:
        print("MAE improvement vs baseline: n/a (baseline MAE is zero)")
    else:
        print(
            "MAE improvement vs baseline: "
            f"{evaluation.improvement_percent:.1f}%"
        )

    saved_plot = save_evaluation_plot(
        evaluation,
        args.plot,
        plot_hours=args.plot_days * HOURS_PER_DAY,
    )
    print(f"Plot saved: {saved_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
