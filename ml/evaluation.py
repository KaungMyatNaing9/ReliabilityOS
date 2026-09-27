"""Evaluation helpers for ReliabilityOS binary failure classifiers.

The functions in this module deliberately avoid an accuracy metric.  In a
predictive-maintenance setting, missing a real failure and alert fatigue are
more consequential than a high overall correct-prediction count.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_KEYS = (
    "recall",
    "precision",
    "f1",
    "false_alarm_rate",
    "roc_auc",
    "brier_score",
    "latency_ms",
)


def _as_1d(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Convert common pandas/numpy inputs into a one-dimensional array."""

    array = np.asarray(values)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array


def infer_positive_label(y: Sequence[Any] | np.ndarray) -> Hashable:
    """Infer the failure class while favouring conventional label ``1``.

    Data profiling normally identifies a binary maintenance target before this
    module is called.  The fallback makes direct use of this module safe for
    boolean and string-labelled examples too.
    """

    values = _as_1d(y)
    unique = list(np.unique(values))
    if not unique:
        raise ValueError("Cannot infer a positive label from an empty target.")
    if 1 in unique:
        return 1
    if True in unique:  # bool targets may not retain integer identity
        return True
    for value in unique:
        if "fail" in str(value).lower() or "fault" in str(value).lower():
            return value
    # Classification targets in the demo are binary.  Sorting gives a stable
    # deterministic fallback when labels are arbitrary strings.
    return sorted(unique, key=lambda item: str(item))[-1]


def _binary_target(
    y: Sequence[Any] | np.ndarray, positive_label: Hashable | None = None
) -> tuple[np.ndarray, Hashable]:
    values = _as_1d(y)
    label = infer_positive_label(values) if positive_label is None else positive_label
    return (values == label).astype(int), label


def calculate_false_alarm_rate(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    positive_label: Hashable | None = None,
) -> float | None:
    """Return FP / (FP + TN), or ``None`` when no normal examples exist."""

    actual, label = _binary_target(y_true, positive_label)
    predicted = (_as_1d(y_pred) == label).astype(int)
    negatives = actual == 0
    denominator = int(negatives.sum())
    if denominator == 0:
        return None
    false_positives = int(((predicted == 1) & negatives).sum())
    return float(false_positives / denominator)


def _positive_probabilities(
    model: Any, features: Any, positive_label: Hashable
) -> np.ndarray | None:
    """Extract the predicted probability for the configured failure class."""

    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = np.asarray(model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            return None
        classes = getattr(model, "classes_", None)
        # Pipelines expose fitted final-estimator attributes inconsistently
        # across sklearn versions, so look through named_steps as well.
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("model"), "classes_", None)
        if classes is not None:
            class_list = list(classes)
            if positive_label in class_list:
                return probabilities[:, class_list.index(positive_label)]
        return probabilities[:, -1]
    except (AttributeError, TypeError, ValueError):
        return None


def evaluate_predictions(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    y_probability: Sequence[float] | np.ndarray | None = None,
    *,
    positive_label: Hashable | None = None,
    latency_ms: float | None = None,
) -> dict[str, float | None]:
    """Calculate ReliabilityOS's industrial evaluation metric contract."""

    actual, label = _binary_target(y_true, positive_label)
    predicted = (_as_1d(y_pred) == label).astype(int)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same number of rows.")

    metrics: dict[str, float | None] = {
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_alarm_rate": calculate_false_alarm_rate(y_true, y_pred, label),
        "roc_auc": None,
        "brier_score": None,
        "latency_ms": None if latency_ms is None else float(latency_ms),
    }

    probabilities = None if y_probability is None else _as_1d(y_probability)
    if probabilities is not None and len(probabilities) == len(actual):
        if len(np.unique(actual)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
            except ValueError:
                pass
            try:
                metrics["brier_score"] = float(brier_score_loss(actual, probabilities))
            except ValueError:
                pass
    return metrics


def evaluate_classifier(
    model: Any,
    features: Any,
    target: Sequence[Any] | np.ndarray,
    *,
    positive_label: Hashable | None = None,
    latency_repeats: int = 3,
) -> dict[str, float | None]:
    """Predict and evaluate a fitted classifier, including median latency.

    Median timing over a few calls produces a useful demo signal while keeping
    the AutoML arena responsive on Streamlit Community Cloud.
    """

    target_values = _as_1d(target)
    label = infer_positive_label(target_values) if positive_label is None else positive_label
    repeats = max(1, min(int(latency_repeats), 5))
    timings: list[float] = []
    predictions: np.ndarray | None = None
    for _ in range(repeats):
        started = perf_counter()
        current = np.asarray(model.predict(features))
        timings.append((perf_counter() - started) * 1000)
        if predictions is None:
            predictions = current
    assert predictions is not None
    probabilities = _positive_probabilities(model, features, label)
    return evaluate_predictions(
        target_values,
        predictions,
        probabilities,
        positive_label=label,
        latency_ms=float(np.median(timings)),
    )


def metric_or_default(metrics: Mapping[str, float | None], key: str, default: float) -> float:
    """Return a finite metric or a documented default for ranking helpers."""

    value = metrics.get(key)
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default
