"""Leakage-safe splitting and preprocessing handoff for ReliabilityOS."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents.profiler import profile_dataset
from ml.features import engineer_features


def temporal_split(
    frame: pd.DataFrame,
    timestamp_column: str | None = None,
    *,
    train_size: float = 0.6,
    validation_size: float = 0.2,
) -> dict[str, Any]:
    """Split data chronologically into non-overlapping train/validation/test.

    When a valid timestamp exists, the latest timestamp in every earlier split
    is less than or equal to the earliest timestamp in the following split. No
    random shuffle is used.  Without a timestamp, source row order is retained
    and the metadata makes that limitation explicit.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("temporal_split expects a pandas DataFrame")
    if len(frame) < 3:
        raise ValueError("At least three rows are required for train/validation/test splitting")
    if not 0 < train_size < 1 or not 0 < validation_size < 1 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than 1")

    ordered = frame.copy()
    ordered["__reliabilityos_original_order"] = range(len(ordered))
    strategy = "ordered"
    parsed: pd.Series | None = None
    if timestamp_column and timestamp_column in ordered.columns:
        parsed = pd.to_datetime(ordered[timestamp_column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            ordered["__reliabilityos_split_time"] = parsed
            ordered = ordered.sort_values(
                ["__reliabilityos_split_time", "__reliabilityos_original_order"],
                kind="stable",
                na_position="last",
            )
            strategy = "temporal"
        else:
            parsed = None

    row_count = len(ordered)
    train_end = max(1, int(row_count * train_size))
    validation_end = max(train_end + 1, int(row_count * (train_size + validation_size)))
    validation_end = min(validation_end, row_count - 1)

    def clean(part: pd.DataFrame) -> pd.DataFrame:
        return part.drop(columns=["__reliabilityos_original_order", "__reliabilityos_split_time"], errors="ignore")

    train = clean(ordered.iloc[:train_end]).copy()
    validation = clean(ordered.iloc[train_end:validation_end]).copy()
    test = clean(ordered.iloc[validation_end:]).copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Split sizes produced an empty partition; provide at least three rows")

    metadata: dict[str, Any] = {
        "strategy": strategy,
        "timestamp_column": timestamp_column if strategy == "temporal" else None,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    if strategy == "temporal" and timestamp_column:
        train_times = pd.to_datetime(train[timestamp_column], errors="coerce")
        validation_times = pd.to_datetime(validation[timestamp_column], errors="coerce")
        test_times = pd.to_datetime(test[timestamp_column], errors="coerce")
        metadata["train_end"] = train_times.max().isoformat() if train_times.notna().any() else None
        metadata["validation_end"] = validation_times.max().isoformat() if validation_times.notna().any() else None
        metadata["test_start"] = test_times.min().isoformat() if test_times.notna().any() else None

    return {"train": train, "validation": validation, "test": test, "metadata": metadata}


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Create an unfitted, train-only sklearn preprocessing transformer."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _usable_feature_columns(
    frame: pd.DataFrame,
    profile: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    excluded = {profile.get("timestamp_column"), profile.get("asset_id_column"), profile.get("target_column")}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            continue
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric.append(str(column))
        else:
            categorical.append(str(column))
    return numeric, categorical


def prepare_dataset(
    frame: pd.DataFrame,
    profile: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Engineer features and provide raw, chronologically pre-split model data.

    The returned X partitions retain pandas column names and are deliberately
    untransformed.  Training code should fit the supplied ``preprocessor`` only
    on ``X_train`` (normally inside an sklearn Pipeline), preventing leakage.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("prepare_dataset expects a pandas DataFrame")
    if frame.empty:
        raise ValueError("Cannot prepare an empty dataset")

    profile_dict = dict(profile) if profile is not None else profile_dataset(frame)
    config = dict(feature_config or {})
    feature_frame = frame.copy()
    if config.get("enable_rolling", True):
        feature_frame = engineer_features(
            feature_frame,
            timestamp_column=profile_dict.get("timestamp_column"),
            asset_id_column=profile_dict.get("asset_id_column"),
            target_column=profile_dict.get("target_column"),
            sensor_columns=config.get("sensor_columns"),
            rolling_windows=config.get("rolling_windows", (3, 6)),
        )

    target_column = profile_dict.get("target_column")
    if target_column and target_column not in feature_frame.columns:
        raise ValueError(f"Profile target column {target_column!r} is not present in data")
    # Supervised models cannot learn from an unknown label.  This filtering is
    # recorded in metadata rather than silently shifting indexes after splitting.
    dropped_unlabelled = 0
    if target_column:
        before = len(feature_frame)
        feature_frame = feature_frame.loc[feature_frame[target_column].notna()].copy()
        dropped_unlabelled = before - len(feature_frame)
    if len(feature_frame) < 3:
        raise ValueError("At least three labelled rows are required after target cleanup")

    numeric_columns, categorical_columns = _usable_feature_columns(feature_frame, profile_dict)
    feature_columns = numeric_columns + categorical_columns
    if not feature_columns:
        raise ValueError("No usable model features remain after schema cleanup")

    split_result = temporal_split(
        feature_frame,
        profile_dict.get("timestamp_column"),
        train_size=float(config.get("train_size", 0.6)),
        validation_size=float(config.get("validation_size", 0.2)),
    )

    def xy(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        x_values = partition.loc[:, feature_columns].copy()
        y_values = partition[target_column].copy() if target_column else None
        return x_values, y_values

    x_train, y_train = xy(split_result["train"])
    x_val, y_val = xy(split_result["validation"])
    x_test, y_test = xy(split_result["test"])
    split_metadata = dict(split_result["metadata"])
    split_metadata["dropped_unlabelled_rows"] = int(dropped_unlabelled)
    split_metadata["feature_engineering"] = {
        "rolling_enabled": bool(config.get("enable_rolling", True)),
        "rolling_windows": list(config.get("rolling_windows", (3, 6))),
    }

    return {
        "X": feature_frame.loc[:, feature_columns].copy(),
        "y": feature_frame[target_column].copy() if target_column else None,
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "features": feature_columns,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": build_preprocessor(numeric_columns, categorical_columns),
        "splits": split_result,
        "split_metadata": split_metadata,
        "profile": profile_dict,
        "feature_frame": feature_frame,
    }


prepare_data = prepare_dataset
split_temporal = temporal_split


__all__ = [
    "build_preprocessor",
    "prepare_data",
    "prepare_dataset",
    "split_temporal",
    "temporal_split",
]
