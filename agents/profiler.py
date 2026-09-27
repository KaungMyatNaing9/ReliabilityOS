"""Dataset inspection for the ReliabilityOS intake workflow.

All functions in this module are deterministic and return plain dictionaries,
which keeps UI rendering and any future orchestration separate from profiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, TextIO

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


_TARGET_NAMES = (
    "failure",
    "failed",
    "failure_flag",
    "machine_failure",
    "breakdown",
    "target",
    "label",
    "y",
    "anomaly",
    "is_anomaly",
)
_TIMESTAMP_NAMES = ("timestamp", "datetime", "date_time", "event_time", "time", "date")
_ASSET_ID_NAMES = (
    "asset_id",
    "machine_id",
    "equipment_id",
    "device_id",
    "unit_id",
    "asset",
    "machine",
)


def load_csv(source: str | Path | BinaryIO | TextIO, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV path or Streamlit-style uploaded file into a DataFrame.

    Column names are trimmed to prevent common upload issues, while all values
    remain untouched for transparent profiling and preprocessing.
    """
    if source is None:
        raise ValueError("A CSV path or uploaded file is required")
    frame = pd.read_csv(source, **read_csv_kwargs)
    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows")
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("CSV column names must be unique after trimming whitespace")
    return frame


def _name_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for normalized_name, original_name in normalized.items():
            if len(candidate) > 1 and candidate in normalized_name:
                return original_name
    return None


def detect_timestamp_column(frame: pd.DataFrame) -> str | None:
    """Infer a timestamp column without treating arbitrary strings as dates."""
    named = _name_match(list(frame.columns), _TIMESTAMP_NAMES)
    if named is not None:
        parsed = pd.to_datetime(frame[named], errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.8:
            return named

    for column in frame.columns:
        series = frame[column]
        if is_datetime64_any_dtype(series):
            return str(column)
    return None


def detect_asset_id_column(frame: pd.DataFrame, timestamp_column: str | None = None) -> str | None:
    """Infer an equipment identifier while avoiding timestamp columns."""
    candidates = [column for column in frame.columns if column != timestamp_column]
    return _name_match(candidates, _ASSET_ID_NAMES)


def detect_target_column(
    frame: pd.DataFrame,
    timestamp_column: str | None = None,
    asset_id_column: str | None = None,
) -> str | None:
    """Find a likely label column using naming and low-cardinality heuristics."""
    excluded = {timestamp_column, asset_id_column}
    available = [str(column) for column in frame.columns if column not in excluded]
    by_name = _name_match(available, _TARGET_NAMES)
    if by_name is not None:
        return by_name

    # A last-column binary indicator is a common uploaded-data convention.
    for column in reversed(available):
        series = frame[column].dropna()
        unique_count = series.nunique()
        if 1 < unique_count <= 2 and (is_bool_dtype(series) or is_numeric_dtype(series)):
            normalized = str(column).lower()
            if any(token in normalized for token in ("flag", "status", "outcome", "class")):
                return column
    return None


def _classification_task(series: pd.Series) -> str | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    unique_count = non_null.nunique()
    is_categorical_label = is_bool_dtype(non_null) or not is_numeric_dtype(non_null)
    if unique_count == 2:
        return "binary_classification"
    if unique_count <= 20 and (is_categorical_label or unique_count <= max(10, len(non_null) * 0.05)):
        return "multiclass_classification"
    return None


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the exact :class:`DatasetProfile` dictionary contract.

    Feature lists intentionally exclude inferred target, timestamp, and asset
    identifiers: those fields are metadata or labels rather than model inputs.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("profile_dataset expects a pandas DataFrame")
    if frame.empty:
        raise ValueError("Cannot profile an empty dataset")

    timestamp_column = detect_timestamp_column(frame)
    asset_id_column = detect_asset_id_column(frame, timestamp_column)
    target_column = detect_target_column(frame, timestamp_column, asset_id_column)
    excluded = {timestamp_column, asset_id_column, target_column}

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if is_numeric_dtype(frame[column]) and not is_bool_dtype(frame[column]):
            numeric_columns.append(str(column))
        else:
            categorical_columns.append(str(column))

    # ``missing_pct`` follows the application-wide fraction convention (e.g.
    # 0.05 means 5%).  UI formatters and the Reliability Critic both consume
    # this directly as a percentage value.
    missing_pct = float(frame.isna().sum().sum() / max(frame.shape[0] * frame.shape[1], 1))
    class_balance: dict[str, dict[str, float | int]] | None = None
    task = _classification_task(frame[target_column]) if target_column else None
    if target_column and task is not None:
        counts = frame[target_column].value_counts(dropna=False)
        total = int(counts.sum())
        class_balance = {
            str(label): {"count": int(count), "pct": round(float(count / total * 100), 3)}
            for label, count in counts.items()
        }

    if task is not None:
        recommended_task = task
    elif timestamp_column and numeric_columns:
        recommended_task = "anomaly_detection_or_forecasting"
    elif numeric_columns:
        recommended_task = "anomaly_detection"
    else:
        recommended_task = "schema_review_required"

    warnings: list[str] = []
    if missing_pct > 0:
        warnings.append(f"Missing values affect {missing_pct:.2%} of cells; imputation will be applied.")
    if timestamp_column is None:
        warnings.append("No timestamp detected; uploaded row order will be used for a deterministic holdout split.")
    if asset_id_column is None:
        warnings.append("No asset identifier detected; rolling features will use one global history.")
    if target_column is None:
        warnings.append("No target label detected; recommend anomaly detection or forecasting review.")
    elif task is None:
        warnings.append("Target is continuous or high-cardinality; classification training is not recommended.")
    if class_balance:
        minority = min(float(entry["pct"]) for entry in class_balance.values())
        if minority < 10:
            warnings.append(f"Severe class imbalance detected: smallest class is {minority:.2f}% of rows.")
    if len(numeric_columns) == 0:
        warnings.append("No numeric sensor columns detected; model quality may be limited.")

    # Keep this literal key set aligned with INTERFACES.md.  Additional facts
    # belong in consumers' own state rather than the shared profile contract.
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "timestamp_column": timestamp_column,
        "asset_id_column": asset_id_column,
        "target_column": target_column,
        "missing_pct": round(missing_pct, 6),
        "class_balance": class_balance,
        "recommended_task": recommended_task,
        "warnings": warnings,
    }


def load_sample_data() -> pd.DataFrame:
    """Load the deterministic bundled predictive-maintenance data set."""
    from data.sample import load_sample_data as _load_sample_data

    return _load_sample_data()


def prepare_dataset(
    frame: pd.DataFrame,
    profile: dict[str, Any] | None = None,
    feature_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile-to-feature-pipeline adapter used by app orchestration.

    Keeping this thin adapter in the agent module gives UI code one natural
    import while the deterministic split and feature logic remains in ``ml``.
    """
    from ml.preprocessing import prepare_dataset as _prepare_dataset

    return _prepare_dataset(frame, profile=profile, feature_config=feature_config)


# Short aliases keep the API pleasant in app code and backwards compatible with
# the terminology used in planning documents.
profile_data = profile_dataset
inspect_dataset = profile_dataset
prepare_data = prepare_dataset


__all__ = [
    "detect_asset_id_column",
    "detect_target_column",
    "detect_timestamp_column",
    "inspect_dataset",
    "load_csv",
    "load_sample_data",
    "prepare_data",
    "prepare_dataset",
    "profile_data",
    "profile_dataset",
]
