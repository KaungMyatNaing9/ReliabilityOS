"""Happy-path contracts for ReliabilityOS drift and challenger helpers."""

from __future__ import annotations

import pandas as pd

from agents.improvement import compare_models, generate_improvement_proposal, train_challenger
from agents.monitor import detect_drift, evaluate_production_performance, simulate_drift
from agents.profiler import load_sample_data, prepare_dataset, profile_dataset
from agents.ml_engineer import select_champion, train_automl
from core.workflow import promotion_gate
from data.sample import generate_sample_dataset


def _profile() -> dict[str, object]:
    return {
        "rows": 100,
        "columns": 6,
        "numeric_columns": ["vibration_mm_s", "torque_nm", "pressure_bar"],
        "categorical_columns": [],
        "timestamp_column": "timestamp",
        "asset_id_column": "asset_id",
        "target_column": "failure",
        "missing_pct": 0.0,
        "class_balance": {"0": 80, "1": 20},
        "recommended_task": "binary_classification",
        "warnings": [],
    }


def _experiment(name: str, recall: float, false_alarm_rate: float) -> dict[str, object]:
    return {
        "experiment_id": name.lower().replace(" ", "-"),
        "model_name": name,
        "model": object(),
        "features": ["vibration_mm_s", "torque_nm"],
        "notes": [],
        "metrics": {
            "recall": recall,
            "precision": 0.70,
            "f1": 0.75,
            "false_alarm_rate": false_alarm_rate,
            "roc_auc": 0.84,
            "brier_score": 0.12,
            "latency_ms": 3.0,
        },
    }


def test_simulated_sensor_drift_preserves_schema_and_is_detected() -> None:
    source = generate_sample_dataset(n_assets=6, periods_per_asset=28)
    profile = _profile()

    production = simulate_drift(source, profile)
    report = detect_drift(source, production, profile)

    assert list(production.columns) == list(source.columns)
    assert production.index.equals(source.index)
    assert production.attrs["reliabilityos_simulation"]["labelled_failure_incidence_uplift"] > 0
    assert report["overall_status"] == "DRIFT"
    assert report["score"] > 0.25
    assert report["features"][0]["severity"] == "HIGH"
    assert {"overall_status", "score", "features"} == set(report)


def test_unchanged_telemetry_is_stable() -> None:
    source = generate_sample_dataset(n_assets=4, periods_per_asset=28)
    report = detect_drift(source, source.copy(), _profile())

    assert report["overall_status"] == "STABLE"
    assert report["score"] == 0.0


def test_proposal_comparison_and_human_gate_are_explicit() -> None:
    champion = _experiment("Champion", recall=0.70, false_alarm_rate=0.14)
    challenger = _experiment("Challenger", recall=0.82, false_alarm_rate=0.09)
    drift_report = {
        "overall_status": "DRIFT",
        "score": 0.42,
        "features": [
            {"feature": "vibration_mm_s", "score": 0.61, "severity": "HIGH"},
            {"feature": "torque_nm", "score": 0.17, "severity": "MEDIUM"},
        ],
    }

    proposal = generate_improvement_proposal(drift_report, champion, _profile())
    comparison = compare_models(champion, challenger)
    pending = promotion_gate(champion, challenger, {"recommendation": "APPROVE_FOR_SHADOW"})
    promoted = promotion_gate(
        champion,
        challenger,
        {"recommendation": "APPROVE_FOR_SHADOW"},
        human_approved=True,
    )

    assert set(proposal) == {"hypotheses", "feature_changes", "model_changes", "reason"}
    assert "vibration_mm_s" in proposal["reason"] or "vibration_mm_s" in " ".join(proposal["hypotheses"])
    assert comparison["recommendation"] == "CHALLENGER_RECOMMENDED"
    assert comparison["human_approval_required"] is True
    assert pending["status"] == "PENDING_HUMAN_APPROVAL"
    assert pending["promoted"] is False
    assert promoted["status"] == "PROMOTED"
    assert promoted["champion"]["model_name"] == "Challenger"


def test_drift_detection_tolerates_extra_non_numeric_schema() -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="h"),
            "asset_id": ["A"] * 12,
            "mode": ["normal"] * 12,
            "sensor": list(range(12)),
            "failure": [0] * 10 + [1, 1],
        }
    )
    production = reference.copy()
    production["sensor"] = production["sensor"] + 8
    report = detect_drift(reference, production, profile=None)

    assert report["overall_status"] == "DRIFT"
    assert [item["feature"] for item in report["features"]] == ["sensor"]


def test_challenger_reuses_the_real_prepared_data_training_path() -> None:
    source = load_sample_data()
    profile = profile_dataset(source)
    prepared = prepare_dataset(source, profile)
    proposal = generate_improvement_proposal(
        {
            "overall_status": "DRIFT",
            "score": 0.3,
            "features": [{"feature": "vibration_mm_s", "score": 0.5, "severity": "HIGH"}],
        }
    )

    challenger = train_challenger(prepared, proposal, profile)

    assert challenger["model"] is not None
    assert challenger["model_name"].startswith("Challenger · ")
    assert "-challenger-" in challenger["experiment_id"]
    assert set(challenger["metrics"]) == {
        "recall",
        "precision",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "brier_score",
        "latency_ms",
    }


def test_labelled_simulation_can_report_deployed_model_degradation_metrics() -> None:
    source = load_sample_data()
    profile = profile_dataset(source)
    champion = select_champion(train_automl(prepare_dataset(source, profile), profile))
    assert champion is not None

    production = simulate_drift(source, profile)
    performance = evaluate_production_performance(champion, production, profile)

    assert performance["labels_available"] is True
    assert performance["rows_evaluated"] > 0
    assert performance["metrics"]["recall"] < champion["metrics"]["recall"]
    assert set(performance["metrics"]) == {
        "recall",
        "precision",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "brier_score",
        "latency_ms",
    }
