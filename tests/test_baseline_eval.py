from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aep_load_forecasting.baseline_eval import (
    evaluate_baselines,
    load_baseline_features,
    main,
)


def baseline_frame(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    target = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "y": target,
            "lag_24": target - 24.0,
            "lag_168": target - 168.0,
        },
        index=index,
    )


def test_evaluate_baselines_uses_exact_trailing_window() -> None:
    test, predictions, results = evaluate_baselines(
        baseline_frame(),
        evaluation_hours=48,
    )

    assert len(test) == 48
    assert test.index.equals(predictions.index)
    assert [result.model for result in results] == [
        "Yesterday (lag_24)",
        "Last week (lag_168)",
        "Blend 50/50",
    ]
    assert [result.mae_mw for result in results] == [24.0, 168.0, 96.0]
    assert [result.rmse_mw for result in results] == [24.0, 168.0, 96.0]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "Datetime,y,lag_24\n2025-01-01,1,1\n",
            "missing required columns",
        ),
        (
            (
                "Datetime,y,lag_24,lag_168\n"
                "not-a-date,1,1,1\n"
            ),
            "invalid timestamp",
        ),
        (
            (
                "Datetime,y,lag_24,lag_168\n"
                "2025-01-01 00:00:00,1,bad,1\n"
            ),
            "non-numeric",
        ),
        (
            (
                "Datetime,y,lag_24,lag_168\n"
                "2025-01-01 00:00:00,1,1,1\n"
                "2025-01-01 02:00:00,2,2,2\n"
            ),
            "consecutive hourly",
        ),
    ],
)
def test_load_baseline_features_rejects_invalid_csv(
    tmp_path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "features.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_baseline_features(path)


def test_command_writes_metrics_and_plot(tmp_path) -> None:
    features_path = tmp_path / "features.csv"
    metrics_path = tmp_path / "nested" / "metrics.csv"
    plot_path = tmp_path / "figures" / "baseline.png"
    frame = baseline_frame()
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
        ]
    )

    assert result == 0
    assert metrics_path.is_file()
    assert plot_path.is_file()
    metrics = pd.read_csv(metrics_path)
    assert metrics["model"].tolist() == [
        "Yesterday (lag_24)",
        "Last week (lag_168)",
        "Blend 50/50",
    ]
