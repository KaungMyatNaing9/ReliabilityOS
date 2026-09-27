"""Dependency-light model evidence for the ReliabilityOS dashboard.

This is intentionally not SHAP: feature-weight evidence is taken directly
from fitted tree importances or linear coefficients, so it stays fast and
available in a Streamlit Cloud-sized demo.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _fitted_parts(experiment: Mapping[str, Any]) -> tuple[Any, Any]:
    pipeline = experiment.get("model")
    steps = getattr(pipeline, "named_steps", {})
    model = steps.get("model", pipeline)
    preprocessor = steps.get("preprocessor")
    return model, preprocessor


def _feature_names(experiment: Mapping[str, Any], preprocessor: Any, count: int) -> list[str]:
    names: list[str] = []
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = [str(name) for name in preprocessor.get_feature_names_out()]
        except (AttributeError, TypeError, ValueError):
            names = []
    if len(names) != count:
        configured = [str(name) for name in experiment.get("features", [])]
        names = configured if len(configured) == count else [f"Transformed feature {index + 1}" for index in range(count)]
    return [name.replace("numeric__", "").replace("categorical__", "") for name in names]


def model_feature_importance(
    experiment: Mapping[str, Any], *, top_n: int = 10
) -> list[dict[str, float | str]]:
    """Return signed-model-agnostic top feature evidence for a fitted result.

    Tree ensembles expose ``feature_importances_``; linear models expose
    ``coef_``.  Other models simply return no evidence rather than making up a
    proxy score.
    """
    if not isinstance(experiment, Mapping) or experiment.get("model") is None:
        return []
    model, preprocessor = _fitted_parts(experiment)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients).reshape(-1) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
    else:
        return []
    if not len(values) or not np.all(np.isfinite(values)):
        return []
    names = _feature_names(experiment, preprocessor, len(values))
    entries = [
        {"feature": name, "importance": round(float(value), 6)}
        for name, value in zip(names, values)
    ]
    return sorted(entries, key=lambda item: (-float(item["importance"]), str(item["feature"])))[: max(1, int(top_n))]


__all__ = ["model_feature_importance"]
