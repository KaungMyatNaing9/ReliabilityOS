"""Improvement-loop coordination with an explicit, non-automatic promotion gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from agents.improvement import generate_improvement_proposal
from agents.monitor import detect_drift, evaluate_production_performance, simulate_drift


def investigate_operating_drift(
    reference_data: pd.DataFrame,
    champion: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create simulated telemetry, evidence, and a proposal without retraining."""

    production_data = simulate_drift(reference_data, profile=profile)
    drift_report = detect_drift(reference_data, production_data, profile=profile)
    proposal = generate_improvement_proposal(drift_report, champion=champion, profile=profile)
    production_performance = (
        evaluate_production_performance(champion, production_data, profile=profile) if champion is not None else None
    )
    return {
        "production_data": production_data,
        "drift_report": drift_report,
        "improvement_proposal": proposal,
        "production_performance": production_performance,
    }


def promotion_gate(
    champion: Mapping[str, Any] | None,
    challenger: Mapping[str, Any] | None,
    challenger_critic: Mapping[str, Any] | None = None,
    *,
    human_approved: bool = False,
) -> dict[str, Any]:
    """Make promotion state explicit; callers must pass a human approval."""

    if challenger is None:
        return {"status": "NO_CHALLENGER", "promoted": False, "champion": champion}
    critic = challenger_critic or {}
    if critic.get("recommendation") == "REJECT":
        return {
            "status": "BLOCKED_BY_CRITIC",
            "promoted": False,
            "champion": champion,
            "reason": "Reliability Critic rejected the challenger.",
        }
    if not human_approved:
        return {
            "status": "PENDING_HUMAN_APPROVAL",
            "promoted": False,
            "champion": champion,
            "reason": "Promotion requires an explicit human approval action.",
        }
    return {
        "status": "PROMOTED",
        "promoted": True,
        "champion": dict(challenger),
        "previous_champion": champion,
        "reason": "Human-approved challenger promotion.",
    }


__all__ = ["investigate_operating_drift", "promotion_gate"]
