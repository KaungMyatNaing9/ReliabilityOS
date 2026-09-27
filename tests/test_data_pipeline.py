from __future__ import annotations

import numpy as np
import pandas as pd

from agents.profiler import profile_dataset
from data.sample import generate_sample_dataset
from ml.features import engineer_features
from ml.preprocessing import prepare_dataset, temporal_split


def test_temporal_split_orders_unsorted_rows_without_future_leakage() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-05", "2025-01-01", "2025-01-06", "2025-01-02", "2025-01-04", "2025-01-03"]
            ),
            "sensor": [5, 1, 6, 2, 4, 3],
            "failure": [0, 0, 1, 0, 1, 0],
        }
    )

    splits = temporal_split(frame, "timestamp", train_size=0.5, validation_size=0.25)

    assert splits["metadata"]["strategy"] == "temporal"
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
    assert splits["metadata"]["train_rows"] == 3
    assert splits["metadata"]["validation_rows"] == 1
    assert splits["metadata"]["test_rows"] == 2


def test_rolling_features_only_use_prior_observations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
            "asset_id": ["A"] * 4,
            "sensor": [10.0, 20.0, 30.0, 40.0],
            "failure": [0, 0, 0, 1],
        }
    )
    baseline = engineer_features(
        frame,
        timestamp_column="timestamp",
        asset_id_column="asset_id",
        target_column="failure",
        sensor_columns=["sensor"],
        rolling_windows=(2,),
    )
    changed_future = frame.copy()
    changed_future.loc[3, "sensor"] = 9_999.0
    changed = engineer_features(
        changed_future,
        timestamp_column="timestamp",
        asset_id_column="asset_id",
        target_column="failure",
        sensor_columns=["sensor"],
        rolling_windows=(2,),
    )

    feature = "sensor__rolling_mean_2"
    assert np.isnan(baseline.loc[0, feature])
    assert baseline.loc[1, feature] == 10.0
    assert baseline.loc[2, feature] == 15.0
    pd.testing.assert_series_equal(baseline.loc[:2, feature], changed.loc[:2, feature])


def test_sample_data_profiles_and_prepares_end_to_end() -> None:
    frame = generate_sample_dataset(n_assets=5, periods_per_asset=24)
    profile = profile_dataset(frame)
    prepared = prepare_dataset(frame, profile)

    assert set(profile) == {
        "rows",
        "columns",
        "numeric_columns",
        "categorical_columns",
        "timestamp_column",
        "asset_id_column",
        "target_column",
        "missing_pct",
        "class_balance",
        "recommended_task",
        "warnings",
    }
    assert profile["recommended_task"] == "binary_classification"
    assert profile["target_column"] not in prepared["features"]
    assert profile["timestamp_column"] not in prepared["features"]
    assert profile["asset_id_column"] not in prepared["features"]
    assert prepared["X_train"].columns.tolist() == prepared["features"]
    assert prepared["split_metadata"]["strategy"] == "temporal"
