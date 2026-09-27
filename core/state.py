"""State helpers that keep Streamlit pages thin and promotion records auditable."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, MutableMapping


DEFAULT_STATE: dict[str, Any] = {
    "profile": None,
    "prepared_data": None,
    "experiments": [],
    "champion": None,
    "critic_report": None,
    "production_data": None,
    "production_metrics": None,
    "drift_report": None,
    "improvement_proposal": None,
    "challenger": None,
    "challenger_critic": None,
    "promotion_history": [],
    "activity_log": [],
}


def ensure_demo_state(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Initialize missing session-state values without overwriting live work."""
    for key, default in DEFAULT_STATE.items():
        if key not in state:
            state[key] = default.copy() if isinstance(default, (dict, list)) else default
    return state


def _experiment_summary(experiment: dict[str, Any] | None) -> dict[str, Any]:
    experiment = experiment or {}
    return {
        "experiment_id": experiment.get("experiment_id"),
        "model_name": experiment.get("model_name"),
        "metrics": dict(experiment.get("metrics") or {}),
        "features": list(experiment.get("features") or []),
    }


def build_promotion_record(
    previous_champion: dict[str, Any] | None,
    promoted_challenger: dict[str, Any],
    critic_report: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe record after an explicit user approval only."""
    if not promoted_challenger:
        raise ValueError("A challenger is required to create a promotion record.")
    return {
        "event_type": "human_approved_promotion",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "from": (previous_champion or {}).get("model_name", "No previous champion"),
        "to": promoted_challenger.get("model_name", "Unnamed challenger"),
        "reason": "Human-approved challenger after drift investigation",
        "critic_status": (critic_report or {}).get("status", "NOT_REVIEWED"),
        "critic_recommendation": (critic_report or {}).get("recommendation", "NOT_REVIEWED"),
        "comparison_recommendation": (comparison or {}).get("recommendation", "NOT_RUN"),
        "previous_champion": _experiment_summary(previous_champion),
        "promoted_challenger": _experiment_summary(promoted_challenger),
    }


def record_human_approved_promotion(
    state: MutableMapping[str, Any],
    previous_champion: dict[str, Any] | None,
    promoted_challenger: dict[str, Any],
    critic_report: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an approval record to session history; never performs promotion itself."""
    record = build_promotion_record(previous_champion, promoted_challenger, critic_report, comparison)
    history = state.setdefault("promotion_history", [])
    history.insert(0, record)
    return record


__all__ = [
    "DEFAULT_STATE",
    "build_promotion_record",
    "ensure_demo_state",
    "record_human_approved_promotion",
]
