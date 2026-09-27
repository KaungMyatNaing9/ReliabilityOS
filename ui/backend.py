"""Thin, guarded adapters from UI actions to ReliabilityOS domain modules.

This module intentionally contains no ML logic.  Its role is to keep pages
usable while a backend module is absent and to normalize small integration
differences into session-state-ready results.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Iterable


class BackendUnavailable(RuntimeError):
    """Raised when a requested domain capability has not landed yet."""


def _callable(module: str, names: Iterable[str]) -> Callable[..., Any]:
    try:
        imported = import_module(module)
    except ImportError as error:
        raise BackendUnavailable(f"{module} is not available yet.") from error
    for name in names:
        candidate = getattr(imported, name, None)
        if callable(candidate):
            return candidate
    raise BackendUnavailable(f"{module} does not expose any of: {', '.join(names)}.")


def load_sample() -> Any:
    return _callable("agents.profiler", ("load_sample_data", "load_demo_data"))()


def load_csv(upload: Any) -> Any:
    return _callable("agents.profiler", ("load_csv", "load_dataset"))(upload)


def profile_data(dataframe: Any) -> dict[str, Any]:
    return _callable("agents.profiler", ("profile_dataset", "profile_data"))(dataframe)


def prepare_data(dataframe: Any, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    return _callable("agents.profiler", ("prepare_dataset", "prepare_data"))(
        dataframe, profile=profile
    )


def train_arena(prepared_data: dict[str, Any], profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    result = _callable("agents.ml_engineer", ("train_automl", "run_automl_arena"))(
        prepared_data, profile=profile
    )
    if isinstance(result, dict) and "experiments" in result:
        return list(result["experiments"])
    return list(result)


def choose_champion(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _callable("agents.ml_engineer", ("select_champion", "choose_champion"))(experiments)
    if isinstance(selected, int):
        return experiments[selected]
    return selected


def model_feature_importance(experiment: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    """Return native model feature-weight evidence when the estimator supports it."""
    return _callable("agents.ml_engineer", ("model_feature_importance",))(experiment, top_n=top_n)


def validate_candidate(
    experiment: dict[str, Any], profile: dict[str, Any] | None = None, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _callable("agents.critic", ("evaluate_candidate", "validate_candidate"))(
        experiment, profile=profile, context=context
    )


def simulate_drift(dataframe: Any, profile: dict[str, Any] | None = None) -> Any:
    return _callable("agents.monitor", ("simulate_drift", "simulate_operating_drift"))(
        dataframe, profile=profile
    )


def detect_drift(reference_data: Any, production_data: Any, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    return _callable("agents.monitor", ("detect_drift", "monitor_drift"))(
        reference_data, production_data, profile=profile
    )


def evaluate_production(experiment: dict[str, Any], production_data: Any, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score labelled simulated telemetry through the monitoring module."""
    return _callable("agents.monitor", ("evaluate_production_performance", "evaluate_production"))(
        experiment, production_data, profile=profile
    )


def create_improvement_proposal(
    drift_report: dict[str, Any], champion: dict[str, Any] | None, profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _callable("agents.improvement", ("generate_improvement_proposal", "propose_improvement"))(
        drift_report, champion=champion, profile=profile
    )


def train_challenger(
    prepared_data: dict[str, Any], proposal: dict[str, Any], profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _callable("agents.improvement", ("train_challenger", "build_challenger"))(
        prepared_data, proposal=proposal, profile=profile
    )


def compare_models(champion: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    return _callable("agents.improvement", ("compare_models", "compare_champion_challenger"))(
        champion, challenger
    )
