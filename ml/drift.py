"""Lightweight, deterministic sensor-distribution drift utilities.

The demo deliberately uses Population Stability Index (PSI) instead of a
heavy monitoring dependency.  PSI is fast, interpretable, and works directly
on the numeric telemetry columns that reach the Streamlit application.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml.evaluation import infer_positive_label


_IDENTIFIER_TOKENS = ("timestamp", "time", "date", "asset_id", "machine_id", "equipment_id", "device_id")
_TARGET_TOKENS = ("failure", "failed", "target", "label", "outcome", "anomaly", "breakdown")
_SENSOR_PRIORITY = (
    "vibration",
    "torque",
    "pressure",
    "wear",
    "temperature",
    "speed",
    "rpm",
    "load",
)


def _profile_exclusions(profile: Mapping[str, Any] | None) -> set[str]:
    if not profile:
        return set()
    return {
        str(value)
        for key in ("timestamp_column", "asset_id_column", "target_column")
        if (value := profile.get(key)) is not None
    }


def _looks_like_metadata(column: str) -> bool:
    normalized = column.lower()
    return any(token in normalized for token in (*_IDENTIFIER_TOKENS, *_TARGET_TOKENS))


def numeric_sensor_columns(
    dataframe: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
    *,
    common_with: pd.DataFrame | None = None,
) -> list[str]:
    """Return valid continuous telemetry columns in stable source order.

    The profile is preferred when supplied, but this remains useful for a CSV
    whose profile has not yet been persisted. Binary maintenance flags and
    identifiers are intentionally excluded from distribution monitoring.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("numeric_sensor_columns expects a pandas DataFrame")
    exclusions = _profile_exclusions(profile)
    configured = (profile or {}).get("numeric_columns") or list(dataframe.columns)
    available = set(dataframe.columns)
    if common_with is not None:
        available &= set(common_with.columns)

    columns: list[str] = []
    for raw_column in configured:
        column = str(raw_column)
        if column not in available or column in exclusions or _looks_like_metadata(column):
            continue
        series = dataframe[column]
        if not is_numeric_dtype(series):
            continue
        finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        # A binary state signal deserves a discrete quality monitor, rather
        # than PSI intended for analogue sensor distributions.
        if finite.nunique() <= 2:
            continue
        columns.append(column)
    return columns


def population_stability_index(
    reference: Sequence[Any] | pd.Series | np.ndarray,
    production: Sequence[Any] | pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins with defensive edge cases."""

    reference_values = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(dtype=float)
    production_values = pd.to_numeric(pd.Series(production), errors="coerce").to_numpy(dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    production_values = production_values[np.isfinite(production_values)]
    if len(reference_values) < 5 or len(production_values) < 5:
        return 0.0

    if np.allclose(reference_values, reference_values[0]):
        # Quantile bins cannot describe a constant baseline.  Make a material
        # operating shift visible without manufacturing an unbounded value.
        scale = max(abs(float(reference_values[0])) * 0.05, 1e-6)
        return 0.0 if np.allclose(production_values, reference_values[0]) else float(
            min(10.0, abs(float(np.nanmedian(production_values)) - float(reference_values[0])) / scale)
        )

    bin_count = max(2, min(int(bins), 20))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_hist, _ = np.histogram(reference_values, bins=edges)
    production_hist, _ = np.histogram(production_values, bins=edges)
    epsilon = 1e-6
    reference_pct = np.maximum(reference_hist / max(reference_hist.sum(), 1), epsilon)
    production_pct = np.maximum(production_hist / max(production_hist.sum(), 1), epsilon)
    return float(np.sum((production_pct - reference_pct) * np.log(production_pct / reference_pct)))


def _severity(score: float) -> str:
    if score >= 0.25:
        return "HIGH"
    if score >= 0.10:
        return "MEDIUM"
    return "LOW"


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared ``DriftReport`` contract for production telemetry."""

    if not isinstance(reference_data, pd.DataFrame) or not isinstance(production_data, pd.DataFrame):
        raise TypeError("detect_drift expects reference and production pandas DataFrames")
    if reference_data.empty or production_data.empty:
        raise ValueError("Drift detection requires non-empty reference and production telemetry")

    features: list[dict[str, Any]] = []
    for column in numeric_sensor_columns(reference_data, profile, common_with=production_data):
        score = population_stability_index(reference_data[column], production_data[column])
        features.append(
            {
                "feature": column,
                "score": round(float(score), 4),
                "severity": _severity(score),
            }
        )
    features.sort(key=lambda item: (-float(item["score"]), str(item["feature"])))
    overall_score = float(np.mean([item["score"] for item in features])) if features else 0.0
    severities = {item["severity"] for item in features}
    if "HIGH" in severities or overall_score >= 0.25:
        overall_status = "DRIFT"
    elif "MEDIUM" in severities or overall_score >= 0.10:
        overall_status = "WARNING"
    else:
        overall_status = "STABLE"
    return {
        "overall_status": overall_status,
        "score": round(overall_score, 4),
        "features": features,
    }


def _priority_key(column: str, position: int) -> tuple[int, int]:
    normalized = column.lower()
    for rank, token in enumerate(_SENSOR_PRIORITY):
        if token in normalized:
            return rank, position
    return len(_SENSOR_PRIORITY), position


def simulate_drift(
    dataframe: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Create deterministic, schema-preserving shifted production telemetry.

    This represents an operating-condition change, not a real PLC feed.  It
    leaves labels intact when present so an evaluator can expose the champion
    model's degraded behavior on the same known incidents.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("simulate_drift expects a pandas DataFrame")
    if dataframe.empty:
        raise ValueError("Cannot simulate drift on an empty dataframe")

    production = dataframe.copy(deep=True)
    candidates = numeric_sensor_columns(dataframe, profile)
    selected = [
        column
        for _, column in sorted((_priority_key(column, index), column) for index, column in enumerate(candidates))[:4]
    ]
    # The values are deliberately deterministic and sizeable enough for PSI
    # to produce a visible DRIFT outcome on the bundled sample data.
    perturbations = ((1.20, 2.20), (1.12, 1.50), (0.82, -1.50), (1.20, 2.00))
    applied: list[str] = []
    for position, column in enumerate(selected):
        original = pd.to_numeric(dataframe[column], errors="coerce")
        finite = original.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) < 5:
            continue
        scale, standard_deviation_shift = perturbations[position % len(perturbations)]
        spread = float(finite.std(ddof=0))
        if not np.isfinite(spread) or spread == 0:
            spread = max(abs(float(finite.median())) * 0.02, 0.01)
        shifted = original * scale + standard_deviation_shift * spread
        # Keep integer sensor schemas integer without changing null positions.
        if is_numeric_dtype(dataframe[column]) and pd.api.types.is_integer_dtype(dataframe[column].dtype):
            shifted = shifted.round().astype(dataframe[column].dtype)
        production[column] = shifted
        applied.append(column)

    target_column = (profile or {}).get("target_column")
    label_flips = 0
    if target_column and target_column in dataframe.columns:
        observed = dataframe[target_column].dropna()
        if observed.nunique() == 2:
            positive_label = infer_positive_label(observed)
            normal_indices = dataframe.index[dataframe[target_column].notna() & (dataframe[target_column] != positive_label)]
            # A shifted operating regime also creates a small, deterministic
            # incidence uplift.  This is explicitly simulation-only and gives
            # the demo a labelled production window where degradation can be
            # measured rather than merely inferred from PSI.
            count = max(1, int(np.ceil(len(normal_indices) * 0.12))) if len(normal_indices) else 0
            if count:
                positions = np.linspace(0, len(normal_indices) - 1, num=count, dtype=int)
                production.loc[normal_indices[positions], target_column] = positive_label
                label_flips = int(count)

    production.attrs = dict(dataframe.attrs)
    production.attrs["reliabilityos_simulation"] = {
        "kind": "deterministic_operating_condition_drift",
        "shifted_features": applied,
        "labelled_failure_incidence_uplift": label_flips,
    }
    return production


calculate_psi = population_stability_index
monitor_drift = detect_drift


__all__ = [
    "calculate_psi",
    "detect_drift",
    "monitor_drift",
    "numeric_sensor_columns",
    "population_stability_index",
    "simulate_drift",
]
