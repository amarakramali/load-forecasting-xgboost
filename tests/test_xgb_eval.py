from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import FORECAST_FEATURES
from src.xgb_eval import (
    evaluate_xgboost,
    load_evaluation_features,
    main,
)


def evaluation_frame(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    target = np.arange(1_000, 1_000 + periods, dtype=float)
    return pd.DataFrame(
        {
            "y": target,
            "hour": index.hour,
            "dayofweek": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "lag_1": target - 1.0,
            "lag_24": target - 24.0,
            "lag_168": target - 168.0,
            "roll_24_mean": target - 12.5,
            "roll_168_mean": target - 84.5,
        },
        index=index,
    )


class LagOneRegressor:
    def __init__(self) -> None:
        self.training_rows = 0
        self.validation_rows = 0

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        eval_set: list[tuple[pd.DataFrame, pd.Series]],
        verbose: bool,
    ) -> LagOneRegressor:
        self.training_rows = len(features)
        self.validation_rows = len(eval_set[0][0])
        assert len(target) == len(features)
        assert verbose is False
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["lag_1"].to_numpy()


def test_evaluate_xgboost_uses_disjoint_exact_windows() -> None:
    model = LagOneRegressor()

    evaluation = evaluate_xgboost(
        evaluation_frame(),
        evaluation_hours=48,
        model=model,
    )

    assert model.training_rows == 104
    assert model.validation_rows == 48
    assert len(evaluation.test) == 48
    assert evaluation.test.index.equals(evaluation.predictions.index)
    assert [result.model for result in evaluation.results] == [
        "Baseline Blend",
        "XGBoost",
    ]
    assert [result.mae_mw for result in evaluation.results] == [96.0, 1.0]
    assert evaluation.improvement_percent == pytest.approx(98.958333)


def test_evaluate_xgboost_rejects_missing_target() -> None:
    with pytest.raises(ValueError, match="target column"):
        evaluate_xgboost(
            evaluation_frame().drop(columns="y"),
            evaluation_hours=48,
            model=LagOneRegressor(),
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "Datetime,y,lag_24\n2025-01-01,1,1\n",
            "missing required columns",
        ),
        (
            (
                "Datetime,y,"
                + ",".join(FORECAST_FEATURES)
                + "\nnot-a-date,"
                + ",".join(["1"] * (len(FORECAST_FEATURES) + 1))
                + "\n"
            ),
            "invalid timestamp",
        ),
        (
            (
                "Datetime,y,"
                + ",".join(FORECAST_FEATURES)
                + "\n2025-01-01 00:00:00,bad,"
                + ",".join(["1"] * len(FORECAST_FEATURES))
                + "\n"
            ),
            "non-numeric",
        ),
        (
            (
                "Datetime,y,"
                + ",".join(FORECAST_FEATURES)
                + "\n2025-01-01 00:00:00,"
                + ",".join(["1"] * (len(FORECAST_FEATURES) + 1))
                + "\n2025-01-01 02:00:00,"
                + ",".join(["2"] * (len(FORECAST_FEATURES) + 1))
                + "\n"
            ),
            "consecutive hourly",
        ),
    ],
)
def test_load_evaluation_features_rejects_invalid_csv(
    tmp_path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "features.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_features(path)


def test_command_writes_metrics_and_plot(tmp_path) -> None:
    features_path = tmp_path / "features.csv"
    metrics_path = tmp_path / "reports" / "metrics.csv"
    plot_path = tmp_path / "figures" / "evaluation.png"
    frame = evaluation_frame()
    frame.index.name = "Datetime"
    frame.to_csv(features_path)

    result = main(
        [
            "--features",
            str(features_path),
            "--metrics",
            str(metrics_path),
            "--plot",
            str(plot_path),
            "--days",
            "2",
            "--plot-days",
            "1",
            "--estimators",
            "5",
        ]
    )

    assert result == 0
    assert metrics_path.is_file()
    assert plot_path.is_file()
    metrics = pd.read_csv(metrics_path)
    assert metrics["model"].tolist() == ["Baseline Blend", "XGBoost"]
