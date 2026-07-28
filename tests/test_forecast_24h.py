from __future__ import annotations

from pathlib import Path

import pandas as pd

import aep_load_forecasting.forecast_24h as command
from aep_load_forecasting.forecasting import FORECAST_FEATURES


def test_command_forwards_estimator_count_and_writes_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    features = pd.DataFrame(
        {
            "y": [100.0] * 25,
            **{name: [1.0] * 25 for name in FORECAST_FEATURES},
        },
        index=pd.date_range("2025-12-01", periods=25, freq="h"),
    )
    history = pd.Series(
        [100.0],
        index=pd.date_range("2026-01-01", periods=1, freq="h"),
    )
    forecast = pd.DataFrame(
        {
            "forecast_xgb_MW": [101.0],
            "baseline_blend_MW": [99.0],
        },
        index=pd.date_range("2026-01-01 01:00", periods=1, freq="h"),
    )
    calibration_model = object()
    final_model = object()
    captured: dict[str, object] = {"training_rows": []}

    monkeypatch.setattr(command, "load_feature_table", lambda _: features)
    monkeypatch.setattr(command, "load_hourly_series", lambda _: history)

    def fake_train_final_model(
        received_features: pd.DataFrame,
        *,
        n_estimators: int,
    ) -> tuple[object, tuple[str, ...]]:
        captured["n_estimators"] = n_estimators
        training_rows = captured["training_rows"]
        assert isinstance(training_rows, list)
        training_rows.append(len(received_features))
        model = calibration_model if len(training_rows) == 1 else final_model
        return model, FORECAST_FEATURES

    monkeypatch.setattr(command, "train_final_model", fake_train_final_model)
    monkeypatch.setattr(
        command,
        "calibrate_recursive_intervals",
        lambda received_model,
        received_history,
        calibration_index,
        columns,
        *,
        horizon,
        coverage: pd.Series([2.0]),
    )
    monkeypatch.setattr(
        command,
        "recursive_forecast",
        lambda received_model,
        received_history,
        columns,
        *,
        horizon: forecast,
    )

    def fake_dump(payload: object, path: str | Path) -> None:
        assert payload == {
            "model": final_model,
            "features": list(FORECAST_FEATURES),
            "prediction_interval": {
                "coverage": 0.9,
                "calibration_days": 1,
                "half_widths_MW": [2.0],
            },
        }
        Path(path).write_bytes(b"model")

    monkeypatch.setattr(command.joblib, "dump", fake_dump)

    def fake_save_plot(
        received_history: pd.Series,
        received_forecast: pd.DataFrame,
        output_path: str | Path,
        *,
        show: bool = False,
    ) -> Path:
        assert received_history is history
        assert received_forecast["forecast_xgb_MW"].tolist() == [101.0]
        assert received_forecast["forecast_xgb_lower_MW"].tolist() == [99.0]
        assert received_forecast["forecast_xgb_upper_MW"].tolist() == [103.0]
        assert show is False
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")
        return path

    monkeypatch.setattr(command, "save_forecast_plot", fake_save_plot)

    model_path = tmp_path / "models" / "model.joblib"
    forecast_path = tmp_path / "reports" / "forecast.csv"
    figure_path = tmp_path / "reports" / "figures" / "forecast.png"
    result = command.main(
        [
            "--input",
            str(tmp_path / "input.csv"),
            "--features",
            str(tmp_path / "features.csv"),
            "--model",
            str(model_path),
            "--output",
            str(forecast_path),
            "--figure",
            str(figure_path),
            "--horizon",
            "1",
            "--estimators",
            "17",
            "--calibration-days",
            "1",
        ]
    )

    assert result == 0
    assert captured["n_estimators"] == 17
    assert captured["training_rows"] == [1, 25]
    assert model_path.is_file()
    assert forecast_path.is_file()
    assert figure_path.is_file()
    saved_forecast = pd.read_csv(forecast_path)
    assert saved_forecast["forecast_xgb_lower_MW"].tolist() == [99.0]
    assert saved_forecast["forecast_xgb_upper_MW"].tolist() == [103.0]
