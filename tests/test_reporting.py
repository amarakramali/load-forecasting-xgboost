from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting import (
    EvaluationResult,
    evaluate_predictions,
    format_result,
    mae_improvement_percent,
    save_results,
)


def test_evaluate_predictions_calculates_expected_metrics() -> None:
    result = evaluate_predictions(
        "Example",
        y_true=[10.0, 20.0, 30.0],
        y_pred=[12.0, 18.0, 33.0],
    )

    assert result == EvaluationResult(
        model="Example",
        mae_mw=pytest.approx(7 / 3),
        rmse_mw=pytest.approx((17 / 3) ** 0.5),
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([1.0], [1.0, 2.0], "same number"),
        ([[1.0]], [[1.0]], "one-dimensional"),
        ([1.0, np.nan], [1.0, 2.0], "finite values"),
    ],
)
def test_evaluate_predictions_rejects_invalid_arrays(
    y_true,
    y_pred,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_predictions("Invalid", y_true, y_pred)


def test_save_results_writes_deterministic_csv(tmp_path) -> None:
    results = [
        EvaluationResult("Baseline", 10.0, 12.5),
        EvaluationResult("XGBoost", 2.0, 3.25),
    ]

    output = save_results(results, tmp_path / "reports" / "metrics.csv")
    saved = pd.read_csv(output)

    assert output.is_file()
    assert saved.to_dict("records") == [
        {"model": "Baseline", "mae_mw": 10.0, "rmse_mw": 12.5},
        {"model": "XGBoost", "mae_mw": 2.0, "rmse_mw": 3.25},
    ]


def test_reporting_helpers_format_and_compare_results() -> None:
    baseline = EvaluationResult("Baseline", 10.0, 12.0)
    candidate = EvaluationResult("Candidate", 2.5, 4.0)

    assert "MAE:    10.00" in format_result(baseline)
    assert mae_improvement_percent(baseline, candidate) == pytest.approx(75.0)


def test_improvement_rejects_zero_reference_mae() -> None:
    reference = EvaluationResult("Perfect", 0.0, 0.0)
    candidate = EvaluationResult("Candidate", 1.0, 1.0)

    with pytest.raises(ValueError, match="greater than zero"):
        mae_improvement_percent(reference, candidate)
