"""Run the complete deterministic demo pipeline with one command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib

from src.baseline_eval import (
    DEFAULT_EVALUATION_DAYS,
    DEFAULT_PLOT_DAYS,
    evaluate_baselines,
    save_baseline_plot,
)
from src.evaluation import HOURS_PER_DAY
from src.forecast_24h import (
    DEFAULT_ESTIMATORS,
    save_forecast_plot,
    train_final_model,
)
from src.forecasting import recursive_forecast
from src.make_features import load_hourly_series, make_feature_table
from src.reporting import save_results
from src.sample_data import (
    DEFAULT_DAYS,
    DEFAULT_SEED,
    DEFAULT_START,
    generate_hourly_load,
    save_sample_data,
)
from src.xgb_eval import evaluate_xgboost, save_evaluation_plot

DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_HORIZON = 24
LAG_HISTORY_HOURS = 168


@dataclass(frozen=True)
class DemoArtifacts:
    """Paths produced by a successful demo pipeline run."""

    source: Path
    features: Path
    baseline_metrics: Path
    baseline_plot: Path
    xgb_metrics: Path
    xgb_plot: Path
    model: Path
    forecast: Path
    forecast_plot: Path

    def paths(self) -> tuple[Path, ...]:
        """Return every generated artifact in pipeline order."""

        return (
            self.source,
            self.features,
            self.baseline_metrics,
            self.baseline_plot,
            self.xgb_metrics,
            self.xgb_plot,
            self.model,
            self.forecast,
            self.forecast_plot,
        )


def demo_artifacts(output_dir: str | Path) -> DemoArtifacts:
    """Build the deterministic artifact layout below one output directory."""

    root = Path(output_dir)
    return DemoArtifacts(
        source=root / "data" / "sample_aep_hourly.csv",
        features=root / "data" / "sample_features_aep.csv",
        baseline_metrics=root / "reports" / "sample_baseline_metrics.csv",
        baseline_plot=(
            root / "reports" / "figures" / "sample_baseline.png"
        ),
        xgb_metrics=root / "reports" / "sample_xgb_metrics.csv",
        xgb_plot=(
            root / "reports" / "figures" / "sample_xgb_evaluation.png"
        ),
        model=root / "models" / "sample_xgb.joblib",
        forecast=root / "reports" / "sample_forecast.csv",
        forecast_plot=(
            root / "reports" / "figures" / "sample_forecast.png"
        ),
    )


def _validate_run_settings(
    *,
    days: int,
    evaluation_days: int,
    plot_days: int,
    horizon: int,
    n_estimators: int,
) -> None:
    settings = {
        "days": days,
        "evaluation_days": evaluation_days,
        "plot_days": plot_days,
        "horizon": horizon,
        "n_estimators": n_estimators,
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if plot_days > evaluation_days:
        raise ValueError("plot_days must not exceed evaluation_days.")

    required_hours = (
        LAG_HISTORY_HOURS
        + 2 * evaluation_days * HOURS_PER_DAY
        + 1
    )
    minimum_days = (required_hours + HOURS_PER_DAY - 1) // HOURS_PER_DAY
    if days < minimum_days:
        raise ValueError(
            f"At least {minimum_days} sample days are required for "
            f"{evaluation_days}-day validation and test windows."
        )


def run_demo_pipeline(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    days: int = DEFAULT_DAYS,
    start: str = DEFAULT_START,
    seed: int = DEFAULT_SEED,
    evaluation_days: int = DEFAULT_EVALUATION_DAYS,
    plot_days: int = DEFAULT_PLOT_DAYS,
    horizon: int = DEFAULT_HORIZON,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> DemoArtifacts:
    """Generate data, evaluate models, and export a future forecast."""

    _validate_run_settings(
        days=days,
        evaluation_days=evaluation_days,
        plot_days=plot_days,
        horizon=horizon,
        n_estimators=n_estimators,
    )
    artifacts = demo_artifacts(output_dir)
    evaluation_hours = evaluation_days * HOURS_PER_DAY
    plot_hours = plot_days * HOURS_PER_DAY

    sample = generate_hourly_load(days=days, start=start, seed=seed)
    save_sample_data(sample, artifacts.source)

    features = make_feature_table(artifacts.source)
    artifacts.features.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(artifacts.features)

    baseline_test, baseline_predictions, baseline_results = (
        evaluate_baselines(features, evaluation_hours=evaluation_hours)
    )
    save_results(baseline_results, artifacts.baseline_metrics)
    save_baseline_plot(
        baseline_test,
        baseline_predictions,
        artifacts.baseline_plot,
        plot_hours=plot_hours,
    )

    xgb_evaluation = evaluate_xgboost(
        features,
        evaluation_hours=evaluation_hours,
        n_estimators=n_estimators,
    )
    save_results(xgb_evaluation.results, artifacts.xgb_metrics)
    save_evaluation_plot(
        xgb_evaluation,
        artifacts.xgb_plot,
        plot_hours=plot_hours,
    )

    history = load_hourly_series(artifacts.source)
    model, feature_columns = train_final_model(
        features,
        n_estimators=n_estimators,
    )
    artifacts.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(feature_columns)},
        artifacts.model,
    )

    forecast = recursive_forecast(
        model,
        history,
        feature_columns,
        horizon=horizon,
    )
    artifacts.forecast.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(artifacts.forecast)
    save_forecast_plot(
        history,
        forecast,
        artifacts.forecast_plot,
    )
    return artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete synthetic load-forecasting demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for data, reports, and models (default: .)",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--evaluation-days",
        type=int,
        default=DEFAULT_EVALUATION_DAYS,
    )
    parser.add_argument("--plot-days", type=int, default=DEFAULT_PLOT_DAYS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--estimators", type=int, default=DEFAULT_ESTIMATORS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = run_demo_pipeline(
        args.output_dir,
        days=args.days,
        start=args.start,
        seed=args.seed,
        evaluation_days=args.evaluation_days,
        plot_days=args.plot_days,
        horizon=args.horizon,
        n_estimators=args.estimators,
    )

    print("Demo pipeline completed:")
    for path in artifacts.paths():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
