from __future__ import annotations

import hashlib
import json

import joblib
import pandas as pd
import pytest

from aep_load_forecasting.demo_pipeline import main, run_demo_pipeline
from aep_load_forecasting.forecasting import FORECAST_FEATURES


def test_command_writes_complete_reproducible_demo(tmp_path) -> None:
    output_dir = tmp_path / "demo"

    result = main(
        [
            "--output-dir",
            str(output_dir),
            "--days",
            "13",
            "--evaluation-days",
            "2",
            "--plot-days",
            "1",
            "--horizon",
            "3",
            "--estimators",
            "5",
        ]
    )

    assert result == 0
    expected = [
        output_dir / "data" / "sample_aep_hourly.csv",
        output_dir / "data" / "sample_features_aep.csv",
        output_dir / "reports" / "sample_baseline_metrics.csv",
        output_dir / "reports" / "figures" / "sample_baseline.png",
        output_dir / "reports" / "sample_xgb_metrics.csv",
        output_dir / "reports" / "figures" / "sample_xgb_evaluation.png",
        output_dir / "models" / "sample_xgb.joblib",
        output_dir / "reports" / "sample_forecast.csv",
        output_dir / "reports" / "figures" / "sample_forecast.png",
        output_dir / "reports" / "sample_run_manifest.json",
    ]
    assert all(path.is_file() for path in expected)

    baseline_metrics = pd.read_csv(expected[2])
    assert baseline_metrics["model"].tolist() == [
        "Yesterday (lag_24)",
        "Last week (lag_168)",
        "Blend 50/50",
    ]
    xgb_metrics = pd.read_csv(expected[4])
    assert xgb_metrics["model"].tolist() == ["Baseline Blend", "XGBoost"]

    model_artifact = joblib.load(expected[6])
    assert tuple(model_artifact["features"]) == FORECAST_FEATURES
    forecast = pd.read_csv(expected[7])
    assert len(forecast) == 3
    assert {
        "forecast_xgb_MW",
        "baseline_blend_MW",
    }.issubset(forecast.columns)

    manifest = json.loads(expected[9].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["parameters"] == {
        "days": 13,
        "start": "2025-01-01 00:00:00",
        "seed": 42,
        "evaluation_days": 2,
        "plot_days": 1,
        "horizon": 3,
        "n_estimators": 5,
    }
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["packages"]["xgboost"]

    artifact_records = manifest["artifacts"]
    expected_artifacts = expected[:9]
    assert set(artifact_records) == {
        path.relative_to(output_dir).as_posix()
        for path in expected_artifacts
    }
    for path in expected_artifacts:
        record = artifact_records[path.relative_to(output_dir).as_posix()]
        contents = path.read_bytes()
        assert record["bytes"] == len(contents)
        assert record["sha256"] == hashlib.sha256(contents).hexdigest()


def test_pipeline_rejects_too_little_history_before_writing(tmp_path) -> None:
    output_dir = tmp_path / "demo"

    with pytest.raises(ValueError, match="At least 12 sample days"):
        run_demo_pipeline(
            output_dir,
            days=11,
            evaluation_days=2,
            plot_days=1,
            n_estimators=5,
        )

    assert not output_dir.exists()
