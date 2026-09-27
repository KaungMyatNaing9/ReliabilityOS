"""Reliability Critic orchestration.

The critic translates deterministic ML checks into the shared ``CriticReport``
contract.  It intentionally contains no UI code and makes no LLM calls: the
deployment decision must be reproducible and inspectable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml.reliability_checks import FAIL, PASS, WARN, run_reliability_checks


CRITICAL_CHECKS = frozenset({"temporal_split_validity", "target_data_leakage"})


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _candidate_name(experiment: Any) -> str:
    """Produce a readable, deterministic candidate name for the report."""
    for key in ("model_name", "candidate", "experiment_id"):
        value = _value(experiment, key)
        if value:
            return str(value)
    return "Unnamed candidate"


def evaluate_candidate(
    experiment: Any,
    profile: Any | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a model candidate against deterministic reliability gates.

    Parameters follow the shared contracts in ``INTERFACES.md``.  The optional
    ``context`` carries split and lineage evidence not represented by the base
    contracts.  Missing non-critical evidence produces a review requirement;
    an explicit temporal breach or target/data leakage rejects the candidate.
    """
    checks = run_reliability_checks(experiment, profile, context)
    critical_failure = any(
        check["status"] == FAIL and check["name"] in CRITICAL_CHECKS for check in checks
    )
    any_warning = any(check["status"] == WARN for check in checks)

    if critical_failure:
        status = FAIL
        recommendation = "REJECT"
    elif any_warning:
        status = WARN
        recommendation = "REVIEW_REQUIRED"
    else:
        status = PASS
        recommendation = "APPROVE_FOR_SHADOW"

    return {
        "candidate": _candidate_name(experiment),
        "status": status,
        "recommendation": recommendation,
        "checks": checks,
    }


def validate_candidate(
    experiment: Any,
    profile: Any | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Backward-compatible name for :func:`evaluate_candidate`."""
    return evaluate_candidate(experiment, profile, context)


__all__ = ["CRITICAL_CHECKS", "evaluate_candidate", "validate_candidate"]
