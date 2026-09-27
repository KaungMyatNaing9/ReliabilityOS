"""Monitoring-agent adapters for simulated production telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ml.drift import detect_drift as _detect_drift
from ml.drift import simulate_drift as _simulate_drift
from ml.evaluation import evaluate_classifier


def simulate_drift(dataframe: pd.DataFrame, profile: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Inject deterministic operating-condition drift into demo telemetry."""

    return _simulate_drift(dataframe, profile=profile)


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure numeric sensor distribution shift using PSI."""

    return _detect_drift(reference_data, production_data, profile=profile)


def evaluate_production_performance(
    experiment: Mapping[str, Any],
    production_data: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a deployed experiment on labelled simulated telemetry.

    The helper is intentionally optional for real production feeds, where
    labels arrive late or not at all.  For the bundled drift simulation it
    rebuilds the same leakage-safe feature frame and evaluates the existing
    fitted pipeline on its preserved temporal holdout.
    """

    if not isinstance(experiment, Mapping):
        raise TypeError("evaluate_production_performance expects an experiment mapping")
    if not isinstance(production_data, pd.DataFrame):
        raise TypeError("evaluate_production_performance expects a pandas DataFrame")
    target_column = (profile or {}).get("target_column")
    if not target_column or target_column not in production_data.columns:
        return {"labels_available": False, "rows_evaluated": 0, "metrics": None}
    model = experiment.get("model")
    if model is None:
        raise ValueError("A fitted deployed model is required for production evaluation")

    from ml.preprocessing import prepare_dataset

    prepared = prepare_dataset(production_data, profile=profile)
    expected_features = list(experiment.get("features") or prepared["features"])
    features = prepared["X_test"].reindex(columns=expected_features)
    target = prepared["y_test"]
    metrics = evaluate_classifier(model, features, target)
    return {"labels_available": True, "rows_evaluated": int(len(features)), "metrics": metrics}


simulate_operating_drift = simulate_drift
monitor_drift = detect_drift
evaluate_production = evaluate_production_performance


__all__ = [
    "detect_drift",
    "evaluate_production",
    "evaluate_production_performance",
    "monitor_drift",
    "simulate_drift",
    "simulate_operating_drift",
]
