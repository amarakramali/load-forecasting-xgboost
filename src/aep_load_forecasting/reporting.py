"""Reusable metric calculation and report persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class EvaluationResult:
    """Regression metrics for one forecast model."""

    model: str
    mae_mw: float
    rmse_mw: float


def evaluate_predictions(
    model: str,
    y_true,
    y_pred,
) -> EvaluationResult:
    """Calculate MAE and RMSE after validating prediction arrays."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)

    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional.")
    if truth.size == 0:
        raise ValueError("Targets and predictions must not be empty.")
    if truth.shape != prediction.shape:
        raise ValueError(
            "Targets and predictions must contain the same number of values."
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and predictions must contain finite values.")

    return EvaluationResult(
        model=model,
        mae_mw=float(mean_absolute_error(truth, prediction)),
        rmse_mw=float(mean_squared_error(truth, prediction) ** 0.5),
    )


def format_result(result: EvaluationResult) -> str:
    """Format a compact terminal summary."""

    return (
        f"{result.model:24s}  MAE: {result.mae_mw:8.2f}   "
        f"RMSE: {result.rmse_mw:8.2f}"
    )


def save_results(
    results: Sequence[EvaluationResult],
    output_path: str | Path,
) -> Path:
    """Persist evaluation results as a deterministic CSV report."""

    if not results:
        raise ValueError("At least one evaluation result is required.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(asdict(result) for result in results)
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def mae_improvement_percent(
    reference: EvaluationResult,
    candidate: EvaluationResult,
) -> float:
    """Return the candidate's percentage MAE improvement over a reference."""

    if reference.mae_mw <= 0:
        raise ValueError("Reference MAE must be greater than zero.")
    return (reference.mae_mw - candidate.mae_mw) / reference.mae_mw * 100
