"""Train the final model and export a recursive load forecast."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from matplotlib.figure import Figure
from xgboost import XGBRegressor

from aep_load_forecasting.forecasting import (
    FORECAST_FEATURES,
    recursive_forecast,
    validate_feature_columns,
)
from aep_load_forecasting.make_features import (
    DEFAULT_INPUT,
    load_hourly_series,
)

DEFAULT_FEATURES = Path("data") / "features_aep.csv"
DEFAULT_FORECAST = Path("reports") / "forecast_next24h.csv"
DEFAULT_FIGURE = Path("reports") / "figures" / "forecast_next24h.png"
DEFAULT_MODEL = Path("models") / "aep_xgb.joblib"
DEFAULT_ESTIMATORS = 800


def load_feature_table(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the feature table used to train the final model."""

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
    if timestamps.isna().any():
        raise ValueError("Feature CSV contains invalid timestamps.")
    if timestamps.duplicated().any():
        raise ValueError("Feature CSV contains duplicate timestamps.")

    ordered_columns = ("y", *FORECAST_FEATURES)
    numeric = frame.loc[:, ordered_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.isna().any().any():
        raise ValueError("Feature CSV contains non-numeric or missing values.")

    numeric.index = pd.DatetimeIndex(timestamps, name="Datetime")
    return numeric.sort_index()


def train_final_model(
    features: pd.DataFrame,
    *,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> tuple[XGBRegressor, tuple[str, ...]]:
    """Fit the production model on all available feature rows."""

    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")
    feature_columns = validate_feature_columns(
        [column for column in features.columns if column != "y"]
    )
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(features.loc[:, feature_columns], features["y"])
    return model, feature_columns


def save_forecast_plot(
    history: pd.Series,
    forecast: pd.DataFrame,
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Save the last seven days of history with the future forecast."""

    path = Path(output_path)
    end = history.index.max()
    last_seven_days = history.loc[end - pd.Timedelta(days=7) : end]

    if show:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
    else:
        figure = Figure()
        axis = figure.subplots()
    axis.plot(
        last_seven_days.index,
        last_seven_days.values,
        label="Actual (last 7 days)",
    )
    axis.plot(
        forecast.index,
        forecast["baseline_blend_MW"],
        label="Baseline (blend)",
    )
    axis.plot(
        forecast.index,
        forecast["forecast_xgb_MW"],
        label="XGBoost forecast",
    )
    axis.set_title(f"AEP Load Forecast: next {len(forecast)} hours")
    axis.set_xlabel("Time")
    axis.set_ylabel("MW")
    axis.legend()
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
        plt.close(figure)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost and export a recursive load forecast."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of future hours to forecast (default: 24)",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=DEFAULT_ESTIMATORS,
        help=f"Number of boosting rounds (default: {DEFAULT_ESTIMATORS})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the forecast plot after saving it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = load_feature_table(args.features)
    history = load_hourly_series(args.input)
    model, feature_columns = train_final_model(
        features,
        n_estimators=args.estimators,
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(feature_columns)},
        args.model,
    )
    print(f"Model saved: {args.model}")

    forecast = recursive_forecast(
        model,
        history,
        feature_columns,
        horizon=args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output)
    print(f"Forecast saved: {args.output}")

    saved_figure = save_forecast_plot(
        history,
        forecast,
        args.figure,
        show=args.show,
    )
    print(f"Plot saved: {saved_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
