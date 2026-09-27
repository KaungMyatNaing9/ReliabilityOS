"""Deterministic reliability checks for predictive-maintenance candidates.

This module deliberately has no Streamlit or model-training dependencies.  It
accepts the shared ``ExperimentResult`` and ``DatasetProfile`` dictionaries,
plus optional split/lineage metadata, so it can be run for both the initial
champion and an improvement challenger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isfinite
import re
from typing import Any


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Values are intentionally conservative for the bundled demo.  Callers can
# override individual values through ``context={"thresholds": {...}}``.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_failure_recall": 0.70,
    "max_false_alarm_rate": 0.15,
    "max_brier_score": 0.20,
    "max_latency_ms": 50.0,
    "missing_pct_warn": 0.05,
    "severe_imbalance_pct": 0.02,
    "imbalance_warn_pct": 0.10,
}


def _get(source: Any, key: str, default: Any = None) -> Any:
    """Read from a mapping or a lightweight result object without raising."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _first(sources: Sequence[Any], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            value = _get(source, key)
            if value is not None:
                return value
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    """Convert an ordinary numeric value while rejecting bools and NaNs."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _ratio(value: Any) -> float | None:
    """Normalize a profile percentage (``5``) or fraction (``.05``)."""
    number = _number(value)
    if number is None or number < 0:
        return None
    return number / 100.0 if number > 1.0 else number


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "valid", "passed", "pass"}:
            return True
        if normalized in {"false", "no", "invalid", "failed", "fail"}:
            return False
    return None


def _check(name: str, status: str, evidence: str) -> dict[str, str]:
    return {"name": name, "status": status, "evidence": evidence}


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _is_classification(experiment: Any, profile: Any) -> bool:
    task = str(_get(profile, "recommended_task", "") or "").lower()
    if "classification" in task or "failure" in task:
        return True
    if _get(profile, "class_balance") is not None:
        return True
    metrics = _as_mapping(_get(experiment, "metrics"))
    return any(
        metrics.get(metric) is not None
        for metric in ("recall", "failure_recall", "false_alarm_rate", "brier_score")
    )


def _thresholds(context: Any) -> dict[str, float]:
    configured = _as_mapping(_get(context, "thresholds"))
    values = DEFAULT_THRESHOLDS.copy()
    aliases = {
        "minimum_failure_recall": "min_failure_recall",
        "failure_recall_min": "min_failure_recall",
        "false_alarm_rate_max": "max_false_alarm_rate",
        "maximum_false_alarm_rate": "max_false_alarm_rate",
        "brier_score_max": "max_brier_score",
        "maximum_brier_score": "max_brier_score",
        "inference_latency_ms_max": "max_latency_ms",
        "maximum_latency_ms": "max_latency_ms",
    }
    for raw_key, raw_value in configured.items():
        key = aliases.get(str(raw_key), str(raw_key))
        value = _ratio(raw_value) if key.endswith("pct") or key.endswith("rate") else _number(raw_value)
        if key in values and value is not None and value >= 0:
            values[key] = value
    return values


def _coerce_features(value: Any) -> list[str]:
    if value is None or isinstance(value, (str, bytes)):
        return [str(value)] if value else []
    if not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _parse_timestamp(value: Any) -> Any:
    """Return a comparable timestamp when possible, preserving native dates."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def _temporal_split_check(experiment: Any, profile: Any, context: Any) -> dict[str, str]:
    timestamp_column = _get(profile, "timestamp_column")
    if not timestamp_column:
        return _check(
            "temporal_split_validity",
            PASS,
            "No timestamp column was detected; a temporal split is not applicable.",
        )

    split = _as_mapping(_first((context, experiment), "split", "split_metadata"))
    sources = (context, experiment, split)
    explicit_valid = _boolean(_first(sources, "temporal_split_valid", "is_valid", "valid"))
    strategy = _first(sources, "split_strategy", "strategy", "method")
    strategy_name = str(strategy or "").strip().lower()
    if explicit_valid is False:
        return _check(
            "temporal_split_validity",
            FAIL,
            "Split metadata explicitly marks temporal evaluation as invalid.",
        )
    if strategy_name in {"random", "random_split", "stratified", "stratified_random"}:
        return _check(
            "temporal_split_validity",
            FAIL,
            f"Timestamp '{timestamp_column}' is present but split strategy is '{strategy}'.",
        )

    train_end = _first(sources, "train_end", "train_max_timestamp", "train_latest_timestamp")
    test_start = _first(sources, "test_start", "test_min_timestamp", "test_earliest_timestamp")
    if train_end is not None and test_start is not None:
        parsed_train_end = _parse_timestamp(train_end)
        parsed_test_start = _parse_timestamp(test_start)
        try:
            if parsed_train_end >= parsed_test_start:
                return _check(
                    "temporal_split_validity",
                    FAIL,
                    "Training data overlaps or follows the evaluation period "
                    f"(train_end={train_end}, test_start={test_start}).",
                )
        except TypeError:
            return _check(
                "temporal_split_validity",
                WARN,
                "Temporal boundary metadata could not be compared; review the split manually.",
            )
        return _check(
            "temporal_split_validity",
            PASS,
            f"Chronological boundary validated: train_end={train_end} before test_start={test_start}.",
        )

    if explicit_valid is True or strategy_name in {
        "temporal",
        "time",
        "time_based",
        "chronological",
    }:
        return _check(
            "temporal_split_validity",
            PASS,
            f"Temporal split is declared valid for timestamp '{timestamp_column}'.",
        )
    return _check(
        "temporal_split_validity",
        WARN,
        f"Timestamp '{timestamp_column}' exists but temporal split metadata was not supplied.",
    )


def _leakage_check(experiment: Any, profile: Any, context: Any) -> dict[str, str]:
    sources = (context, experiment, profile)
    explicit = _boolean(_first(sources, "leakage_detected", "has_leakage", "target_leakage"))
    if explicit is True:
        return _check("target_data_leakage", FAIL, "Metadata explicitly reports target/data leakage.")

    flagged = _first(sources, "leakage_features", "leaking_features")
    flagged_features = _coerce_features(flagged)
    if flagged_features:
        return _check(
            "target_data_leakage",
            FAIL,
            "Metadata flags potentially leaking features: " + ", ".join(sorted(flagged_features)) + ".",
        )

    target = _first((profile, experiment), "target_column", "target")
    target_key = _normalize_name(target) if target else ""
    features = _coerce_features(_get(experiment, "features"))
    direct_target_features = [feature for feature in features if _normalize_name(feature) == target_key and target_key]
    if direct_target_features:
        return _check(
            "target_data_leakage",
            FAIL,
            "Target column is included in the model features: " + ", ".join(sorted(direct_target_features)) + ".",
        )

    lineage = _as_mapping(_first((context, experiment), "feature_lineage", "lineage"))
    lineage_leaks = [
        str(feature)
        for feature, metadata in lineage.items()
        if _boolean(_first((metadata,), "is_leakage", "leakage_detected", "uses_future_data")) is True
    ]
    if lineage_leaks:
        return _check(
            "target_data_leakage",
            FAIL,
            "Feature lineage identifies future/target leakage in: " + ", ".join(sorted(lineage_leaks)) + ".",
        )

    suspicious_markers = ("future", "postfailure", "afterfailure", "targetencoded", "labelencoded", "outcome")
    suspicious = [
        feature
        for feature in features
        if any(marker in _normalize_name(feature) for marker in suspicious_markers)
    ]
    warning_text = " ".join(str(item) for item in _coerce_features(_get(profile, "warnings"))).lower()
    if _boolean(_first(sources, "leakage_suspected")) is True or "leakage" in warning_text or suspicious:
        detail = (
            "Suspicious feature names require lineage review: " + ", ".join(sorted(suspicious)) + "."
            if suspicious
            else "Available metadata signals possible leakage; review feature lineage."
        )
        return _check("target_data_leakage", WARN, detail)
    return _check(
        "target_data_leakage",
        PASS,
        "No target feature, explicit leakage signal, or suspicious lineage marker was found.",
    )


def _balance_values(class_balance: Any) -> list[float]:
    if not isinstance(class_balance, Mapping):
        return []
    values: list[float] = []
    for value in class_balance.values():
        if isinstance(value, Mapping):
            value = _first((value,), "count", "rows", "proportion", "pct", "percentage")
        number = _number(value)
        if number is not None and number >= 0:
            values.append(number)
    return values


def _class_imbalance_check(experiment: Any, profile: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    if not _is_classification(experiment, profile):
        return _check("class_imbalance", PASS, "Not applicable: candidate is not classified as a classification task.")
    values = _balance_values(_get(profile, "class_balance"))
    total = sum(values)
    if len(values) < 2 or total <= 0:
        return _check("class_imbalance", WARN, "Class balance is unavailable; minority failure coverage cannot be verified.")
    minority_pct = min(values) / total
    if minority_pct < thresholds["severe_imbalance_pct"]:
        return _check(
            "class_imbalance",
            WARN,
            f"Severe class imbalance: minority class is {minority_pct:.1%} (below {thresholds['severe_imbalance_pct']:.1%}).",
        )
    if minority_pct < thresholds["imbalance_warn_pct"]:
        return _check(
            "class_imbalance",
            WARN,
            f"Class imbalance: minority class is {minority_pct:.1%} (below {thresholds['imbalance_warn_pct']:.1%}).",
        )
    return _check("class_imbalance", PASS, f"Minority class representation is {minority_pct:.1%}.")


def _missing_data_check(profile: Any, context: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    data_quality = _as_mapping(_get(context, "data_quality"))
    raw_missing = _first((context, data_quality, profile), "missing_pct", "missing_percentage")
    missing_pct = _ratio(raw_missing)
    if missing_pct is None:
        return _check("data_quality_missing_values", WARN, "Missing-value rate was not provided in the data profile.")
    if missing_pct <= thresholds["missing_pct_warn"]:
        return _check("data_quality_missing_values", PASS, f"Missing values account for {missing_pct:.1%} of profiled cells.")
    return _check(
        "data_quality_missing_values",
        WARN,
        f"Missing values account for {missing_pct:.1%} of profiled cells (warning threshold {thresholds['missing_pct_warn']:.1%}).",
    )


def _metric_value(experiment: Any, *keys: str) -> float | None:
    metrics = _as_mapping(_get(experiment, "metrics"))
    return _number(_first((metrics, experiment), *keys))


def _calibration_check(experiment: Any, profile: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    if not _is_classification(experiment, profile):
        return _check("calibration_brier_score", PASS, "Not applicable: calibration is a classification check.")
    brier = _metric_value(experiment, "brier_score", "brier")
    if brier is None:
        return _check("calibration_brier_score", WARN, "Brier score is unavailable; calibration cannot be verified.")
    if brier <= thresholds["max_brier_score"]:
        return _check(
            "calibration_brier_score",
            PASS,
            f"Brier score {brier:.3f} meets the ≤ {thresholds['max_brier_score']:.3f} threshold.",
        )
    return _check(
        "calibration_brier_score",
        WARN,
        f"Brier score {brier:.3f} exceeds the ≤ {thresholds['max_brier_score']:.3f} threshold.",
    )


def _recall_check(experiment: Any, profile: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    if not _is_classification(experiment, profile):
        return _check("failure_recall", PASS, "Not applicable: failure recall is a classification check.")
    recall = _metric_value(experiment, "recall", "failure_recall")
    if recall is None:
        return _check("failure_recall", WARN, "Failure recall is unavailable; missed-failure risk cannot be verified.")
    if recall >= thresholds["min_failure_recall"]:
        return _check(
            "failure_recall",
            PASS,
            f"Failure recall {recall:.1%} meets the ≥ {thresholds['min_failure_recall']:.1%} threshold.",
        )
    return _check(
        "failure_recall",
        WARN,
        f"Failure recall {recall:.1%} is below the ≥ {thresholds['min_failure_recall']:.1%} threshold.",
    )


def _false_alarm_check(experiment: Any, profile: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    if not _is_classification(experiment, profile):
        return _check("false_alarm_rate", PASS, "Not applicable: false-alarm rate is a classification check.")
    false_alarm_rate = _metric_value(experiment, "false_alarm_rate", "far")
    if false_alarm_rate is None:
        return _check("false_alarm_rate", WARN, "False-alarm rate is unavailable; operational alert burden cannot be verified.")
    if false_alarm_rate <= thresholds["max_false_alarm_rate"]:
        return _check(
            "false_alarm_rate",
            PASS,
            f"False-alarm rate {false_alarm_rate:.1%} meets the ≤ {thresholds['max_false_alarm_rate']:.1%} threshold.",
        )
    return _check(
        "false_alarm_rate",
        WARN,
        f"False-alarm rate {false_alarm_rate:.1%} exceeds the ≤ {thresholds['max_false_alarm_rate']:.1%} threshold.",
    )


def _latency_check(experiment: Any, thresholds: Mapping[str, float]) -> dict[str, str]:
    latency_ms = _metric_value(experiment, "latency_ms", "inference_latency_ms")
    if latency_ms is None:
        return _check("inference_latency", WARN, "Inference latency is unavailable; serving responsiveness cannot be verified.")
    if latency_ms <= thresholds["max_latency_ms"]:
        return _check(
            "inference_latency",
            PASS,
            f"Inference latency {latency_ms:.2f} ms meets the ≤ {thresholds['max_latency_ms']:.2f} ms threshold.",
        )
    return _check(
        "inference_latency",
        WARN,
        f"Inference latency {latency_ms:.2f} ms exceeds the ≤ {thresholds['max_latency_ms']:.2f} ms threshold.",
    )


def run_reliability_checks(
    experiment: Any,
    profile: Any | None = None,
    context: Any | None = None,
) -> list[dict[str, str]]:
    """Run the eight required deterministic checks in a stable display order.

    ``context`` is optional and supports ``split``/``split_metadata``, direct
    temporal fields, ``feature_lineage``, ``data_quality``, and a
    ``thresholds`` dictionary.  Unknown fields are ignored, making the
    function safe while the upstream modules are developed in parallel.
    """
    profile = profile or {}
    context = context or {}
    thresholds = _thresholds(context)
    return [
        _temporal_split_check(experiment, profile, context),
        _leakage_check(experiment, profile, context),
        _class_imbalance_check(experiment, profile, thresholds),
        _missing_data_check(profile, context, thresholds),
        _calibration_check(experiment, profile, thresholds),
        _recall_check(experiment, profile, thresholds),
        _false_alarm_check(experiment, profile, thresholds),
        _latency_check(experiment, thresholds),
    ]
