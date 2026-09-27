"""Focused contract tests for the deterministic Reliability Critic."""

from __future__ import annotations

from agents.critic import evaluate_candidate, validate_candidate


PROFILE = {
    "rows": 1_000,
    "columns": 8,
    "numeric_columns": ["air_temperature", "process_temperature"],
    "categorical_columns": ["machine_type"],
    "timestamp_column": "event_time",
    "asset_id_column": "machine_id",
    "target_column": "failure",
    "missing_pct": 0.01,
    "class_balance": {"0": 900, "1": 100},
    "recommended_task": "binary_classification",
    "warnings": [],
}


PASSING_EXPERIMENT = {
    "experiment_id": "exp-good",
    "model_name": "Random Forest",
    "features": ["air_temperature", "process_temperature", "machine_type"],
    "metrics": {
        "recall": 0.82,
        "precision": 0.71,
        "f1": 0.76,
        "false_alarm_rate": 0.08,
        "roc_auc": 0.91,
        "brier_score": 0.12,
        "latency_ms": 4.5,
    },
}


VALID_CONTEXT = {
    "split": {
        "strategy": "temporal",
        "train_end": "2025-01-31T23:59:59",
        "test_start": "2025-02-01T00:00:00",
    }
}


def test_passing_candidate_is_approved_for_shadow() -> None:
    report = evaluate_candidate(PASSING_EXPERIMENT, PROFILE, VALID_CONTEXT)

    assert report["candidate"] == "Random Forest"
    assert report["status"] == "PASS"
    assert report["recommendation"] == "APPROVE_FOR_SHADOW"
    assert len(report["checks"]) == 8
    assert {check["status"] for check in report["checks"]} == {"PASS"}
    assert {"name", "status", "evidence"} == set(report["checks"][0])


def test_leaking_target_is_rejected() -> None:
    leaking = {**PASSING_EXPERIMENT, "features": ["air_temperature", "failure"]}
    report = validate_candidate(leaking, PROFILE, VALID_CONTEXT)

    assert report["status"] == "FAIL"
    assert report["recommendation"] == "REJECT"
    leakage = next(check for check in report["checks"] if check["name"] == "target_data_leakage")
    assert leakage["status"] == "FAIL"
    assert "Target column" in leakage["evidence"]


def test_invalid_temporal_split_is_rejected() -> None:
    invalid_context = {"split": {"strategy": "temporal", "train_end": "2025-02-01", "test_start": "2025-01-31"}}
    report = evaluate_candidate(PASSING_EXPERIMENT, PROFILE, invalid_context)

    assert report["status"] == "FAIL"
    temporal = next(check for check in report["checks"] if check["name"] == "temporal_split_validity")
    assert temporal["status"] == "FAIL"


def test_noncritical_threshold_miss_requires_review_not_rejection() -> None:
    degraded = {
        **PASSING_EXPERIMENT,
        "metrics": {**PASSING_EXPERIMENT["metrics"], "recall": 0.5, "false_alarm_rate": 0.22},
    }
    report = evaluate_candidate(degraded, PROFILE, VALID_CONTEXT)

    assert report["status"] == "WARN"
    assert report["recommendation"] == "REVIEW_REQUIRED"
