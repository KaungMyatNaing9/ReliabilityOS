"""Agent-facing orchestration for deterministic ReliabilityOS AutoML runs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ml.training import select_champion as _select_champion
from ml.training import train_automl as _train_automl
from ml.training import weighted_selection_score
from ml.explain import model_feature_importance


def training_plan(profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a transparent plan suitable for the Autonomous Activity feed."""

    profile = profile or {}
    timestamp = profile.get("timestamp_column")
    return {
        "agent": "ML Engineer",
        "task": "binary_failure_classification",
        "candidates": ["Logistic Regression", "Random Forest", "XGBoost or sklearn fallback"],
        "split_strategy": "supplied temporal split" if timestamp else "supplied prepared split",
        "selection": "50% failure recall with false-alarm and calibration penalties",
    }


def train_automl(
    prepared_data: Mapping[str, Any], profile: Mapping[str, Any] | None = None, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run the real, bounded AutoML arena and return ``ExperimentResult`` rows."""

    return _train_automl(prepared_data, profile, **kwargs)


def select_champion(experiments: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select the recall-forward champion from AutoML experiment results."""

    return _select_champion(experiments)


__all__ = [
    "model_feature_importance",
    "select_champion",
    "train_automl",
    "training_plan",
    "weighted_selection_score",
]
