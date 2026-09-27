"""Leakage-safe, lightweight predictive-maintenance feature engineering."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype


_IDENTIFIER_TOKENS = ("timestamp", "time", "date", "asset_id", "machine_id", "equipment_id", "device_id")
_TARGET_TOKENS = ("failure", "target", "label", "anomaly", "breakdown")


def _default_sensor_columns(
    frame: pd.DataFrame,
    timestamp_column: str | None,
    asset_id_column: str | None,
    target_column: str | None,
) -> list[str]:
    excluded = {timestamp_column, asset_id_column, target_column}
    sensors: list[str] = []
    for column in frame.columns:
        normalized = str(column).lower()
        if column in excluded or any(token in normalized for token in _TARGET_TOKENS):
            continue
        if any(token in normalized for token in _IDENTIFIER_TOKENS):
            continue
        if is_numeric_dtype(frame[column]):
            sensors.append(str(column))
    return sensors


def engineer_features(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    asset_id_column: str | None = None,
    target_column: str | None = None,
    sensor_columns: Iterable[str] | None = None,
    rolling_windows: Iterable[int] = (3, 6),
) -> pd.DataFrame:
    """Return a copy with backward-looking sensor features.

    The frame is ordered by asset and timestamp (or its existing row order when
    no timestamp is available).  Every rolling average is computed after a
    one-row shift, so the feature for a row never includes its current or any
    future readings.  Current-minus-previous deltas are also safe because they
    rely exclusively on the preceding observation for that asset.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("engineer_features expects a pandas DataFrame")
    if frame.empty:
        return frame.copy()

    windows = tuple(sorted({int(window) for window in rolling_windows if int(window) >= 2}))
    if sensor_columns is None:
        sensors = _default_sensor_columns(frame, timestamp_column, asset_id_column, target_column)
    else:
        sensors = [str(column) for column in sensor_columns if column in frame.columns and is_numeric_dtype(frame[column])]

    result = frame.copy()
    result["__reliabilityos_original_order"] = range(len(result))
    if timestamp_column and timestamp_column in result.columns:
        parsed_timestamp = pd.to_datetime(result[timestamp_column], errors="coerce")
        result["__reliabilityos_sort_time"] = parsed_timestamp
        sort_columns = []
        if asset_id_column and asset_id_column in result.columns:
            sort_columns.append(asset_id_column)
        sort_columns.extend(["__reliabilityos_sort_time", "__reliabilityos_original_order"])
        result = result.sort_values(sort_columns, kind="stable", na_position="last")
    elif asset_id_column and asset_id_column in result.columns:
        result = result.sort_values([asset_id_column, "__reliabilityos_original_order"], kind="stable")

    if asset_id_column and asset_id_column in result.columns:
        grouping_key = result[asset_id_column].fillna("__unknown_asset__").astype(str)
    else:
        grouping_key = pd.Series("__all_assets__", index=result.index)

    for column in sensors:
        history = result[column].groupby(grouping_key, sort=False).shift(1)
        result[f"{column}__delta_1"] = result[column] - history
        for window in windows:
            # Grouping the shifted history avoids a cross-asset rolling window.
            result[f"{column}__rolling_mean_{window}"] = history.groupby(grouping_key, sort=False).transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )

    # A few domain-readable current-observation features add signal without
    # using labels or future data.  They are included only when schema permits.
    if {"process_temperature_k", "air_temperature_k"}.issubset(result.columns):
        result["temperature_delta_k"] = result["process_temperature_k"] - result["air_temperature_k"]
    if {"rotational_speed_rpm", "torque_nm"}.issubset(result.columns):
        result["mechanical_power_proxy"] = result["rotational_speed_rpm"] * result["torque_nm"] / 1000.0

    result = result.sort_values("__reliabilityos_original_order", kind="stable")
    return result.drop(columns=["__reliabilityos_original_order", "__reliabilityos_sort_time"], errors="ignore")


build_feature_frame = engineer_features


__all__ = ["build_feature_frame", "engineer_features"]
