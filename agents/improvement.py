"""Evidence-led, deterministic challenger experiments for ReliabilityOS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha1
from typing import Any

from agents.ml_engineer import select_champion, train_automl, weighted_selection_score
from ml.evaluation import metric_or_default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metric(value: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    return metric_or_default(_as_mapping(value.get("metrics")), name, default)


def _drifted_features(drift_report: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    report = _as_mapping(drift_report)
    items = report.get("features", [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []
    valid = [item for item in items if isinstance(item, Mapping) and item.get("feature")]
    return sorted(valid, key=lambda item: (-float(item.get("score") or 0.0), str(item["feature"])))


def generate_improvement_proposal(
    drift_report: Mapping[str, Any],
    champion: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate drift evidence and champion weaknesses into a shared proposal.

    The proposal is deliberately descriptive and reproducible.  It does not
    alter a deployed model, auto-promote a candidate, or make an LLM claim.
    """

    del profile  # The profile is accepted as public workflow context.
    report = _as_mapping(drift_report)
    highest = _drifted_features(report)[:3]
    feature_names = [str(item["feature"]) for item in highest]
    drift_status = str(report.get("overall_status", "STABLE"))
    champion_metrics = _as_mapping(_as_mapping(champion).get("metrics"))

    hypotheses: list[str] = []
    feature_changes: list[str] = []
    if feature_names:
        joined = ", ".join(feature_names)
        hypotheses.append(f"Operating conditions shifted most in {joined}, so the champion's learned sensor ranges may be stale.")
        for feature in feature_names:
            feature_changes.append(f"Re-evaluate lag and rolling summaries for {feature} under the shifted operating regime.")
    else:
        hypotheses.append("No numeric drift feature was available; verify telemetry coverage before changing the predictive pipeline.")
        feature_changes.append("Preserve the current feature schema and collect a new labelled operating window for review.")

    recall = metric_or_default(champion_metrics, "recall", 0.0)
    false_alarm_rate = metric_or_default(champion_metrics, "false_alarm_rate", 1.0)
    brier_score = metric_or_default(champion_metrics, "brier_score", 0.5)
    if champion_metrics:
        if recall < 0.75:
            hypotheses.append(f"Champion failure recall ({recall:.1%}) is below the demo reliability target; prioritize catching missed failures.")
        if false_alarm_rate > 0.15:
            hypotheses.append(f"Champion false-alarm rate ({false_alarm_rate:.1%}) risks alert fatigue under the new regime.")
        if brier_score > 0.20:
            hypotheses.append(f"Champion probability calibration (Brier {brier_score:.3f}) needs review after the distribution shift.")

    model_changes = [
        "Run a bounded challenger arena with a deeper, larger tree ensemble and class-imbalance-aware training.",
        "Retune probability-sensitive candidates while preserving the original temporal holdout split.",
    ]
    reason = (
        f"{drift_status.title()} evidence triggered a deterministic challenger experiment; "
        "the candidate will require Reliability Critic review and explicit human approval before promotion."
    )
    return {
        "hypotheses": hypotheses,
        "feature_changes": feature_changes,
        "model_changes": model_changes,
        "reason": reason,
    }


def _challenger_candidate_config(proposal: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return safe parameter changes understood by the shared AutoML trainer."""

    # Keep the proposal accepted even when called directly by another
    # integration.  The fixed config is intentionally small and bounded for
    # Streamlit Community Cloud responsiveness.
    del proposal
    return {
        "logistic_regression": {"C": 0.7, "max_iter": 450},
        "random_forest": {"n_estimators": 180, "max_depth": 12, "min_samples_leaf": 1},
        "xgboost": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.06},
        "gradient_boosting_fallback": {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.06},
    }


def train_challenger(
    prepared_data: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and select one changed challenger via the shared AutoML arena.

    No training/evaluation logic is duplicated here: the same temporal splits,
    preprocessing pipelines, industrial metrics, and champion selection rules
    are used for both the original arena and its challenger.
    """

    if not isinstance(prepared_data, Mapping):
        raise TypeError("train_challenger expects the prepared-data mapping from the intake pipeline")
    normalized_proposal = _as_mapping(proposal)
    experiments = train_automl(
        prepared_data,
        profile=profile,
        random_state=73,
        candidate_config=_challenger_candidate_config(normalized_proposal),
    )
    selected = select_champion(experiments)
    if selected is None:
        raise RuntimeError("No challenger model trained successfully; inspect the AutoML arena notes.")

    challenger = dict(selected)
    proposal_fingerprint = sha1(repr(sorted(normalized_proposal.items())).encode()).hexdigest()[:8]
    challenger["experiment_id"] = f"{selected.get('experiment_id', 'challenger')}-challenger-{proposal_fingerprint}"
    base_name = str(selected.get("model_name") or "AutoML candidate")
    challenger["model_name"] = f"Challenger · {base_name}"
    challenger["notes"] = [
        *list(selected.get("notes") or []),
        "Challenger was trained with deterministic drift-response parameter changes on the original temporal split.",
        f"Proposal fingerprint: {proposal_fingerprint}.",
    ]
    return challenger


def _difference(challenger_value: Any, champion_value: Any) -> float | None:
    try:
        if challenger_value is None or champion_value is None:
            return None
        return round(float(challenger_value) - float(champion_value), 4)
    except (TypeError, ValueError):
        return None


def compare_models(champion: Mapping[str, Any], challenger: Mapping[str, Any]) -> dict[str, Any]:
    """Return transparent comparison evidence; it never performs promotion."""

    if not isinstance(champion, Mapping) or not isinstance(challenger, Mapping):
        raise TypeError("compare_models expects champion and challenger experiment mappings")
    champion_metrics = _as_mapping(champion.get("metrics"))
    challenger_metrics = _as_mapping(challenger.get("metrics"))
    metric_deltas = {
        metric: _difference(challenger_metrics.get(metric), champion_metrics.get(metric))
        for metric in ("recall", "precision", "f1", "false_alarm_rate", "roc_auc", "brier_score", "latency_ms")
    }
    champion_score = weighted_selection_score(champion_metrics)
    challenger_score = weighted_selection_score(challenger_metrics)
    recall_delta = metric_deltas["recall"] or 0.0
    false_alarm_delta = metric_deltas["false_alarm_rate"] or 0.0
    # A recommendation is merely evidence for a person.  This conservative
    # gate avoids recommending a higher-scoring model that catches fewer
    # failures or materially increases alert fatigue.
    if challenger_score > champion_score and recall_delta >= -0.005 and false_alarm_delta <= 0.02:
        recommendation = "CHALLENGER_RECOMMENDED"
        reason = "The challenger improves the recall-forward reliability score without a material recall or false-alarm regression."
    elif challenger_score < champion_score:
        recommendation = "CHAMPION_RETAINED"
        reason = "The challenger did not improve the recall-forward reliability score on the preserved holdout split."
    else:
        recommendation = "REVIEW_REQUIRED"
        reason = "Metric trade-offs are mixed; a reliability owner should inspect critic evidence before any promotion."
    return {
        "champion": str(champion.get("model_name") or champion.get("experiment_id") or "Champion"),
        "challenger": str(challenger.get("model_name") or challenger.get("experiment_id") or "Challenger"),
        "champion_selection_score": round(champion_score, 4),
        "challenger_selection_score": round(challenger_score, 4),
        "metric_deltas": metric_deltas,
        "recommendation": recommendation,
        "reason": reason,
        "human_approval_required": True,
    }


propose_improvement = generate_improvement_proposal
build_challenger = train_challenger
compare_champion_challenger = compare_models


__all__ = [
    "build_challenger",
    "compare_champion_challenger",
    "compare_models",
    "generate_improvement_proposal",
    "propose_improvement",
    "train_challenger",
]
