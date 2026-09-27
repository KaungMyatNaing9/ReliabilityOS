"""Deterministic predictive-maintenance sample data used by the demo.

The application deliberately ships with a generated-once CSV rather than
depending on an external data service.  ``generate_sample_dataset`` remains
public so tests and demonstrations can create a smaller, reproducible fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_DATA_PATH = Path(__file__).with_name("sample_predictive_maintenance.csv")
_MACHINE_TYPES = np.array(["CNC", "Hydraulic press", "Conveyor"])


def generate_sample_dataset(
    n_assets: int = 18,
    periods_per_asset: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible, compact predictive-maintenance data set.

    Sensor readings progress per asset over time and the failure label is
    generated from observable degradation signals.  It is intentionally a
    simulation, suitable for a fast product demo but not a real factory model.
    """
    if n_assets < 1 or periods_per_asset < 8:
        raise ValueError("n_assets must be positive and periods_per_asset must be at least 8")

    rng = np.random.default_rng(seed)
    asset_numbers = np.repeat(np.arange(1, n_assets + 1), periods_per_asset)
    cycle = np.tile(np.arange(periods_per_asset), n_assets)
    machine_index = (asset_numbers - 1) % len(_MACHINE_TYPES)
    asset_bias = (asset_numbers - 1) / max(n_assets - 1, 1)

    timestamps = pd.Timestamp("2025-01-01 00:00:00") + pd.to_timedelta(cycle * 6, unit="h")
    machine_type = _MACHINE_TYPES[machine_index]
    ambient_temp = 23.0 + 3.3 * np.sin(cycle / 8.0) + rng.normal(0, 0.75, len(cycle))
    air_temperature = 298.0 + ambient_temp / 5.0 + rng.normal(0, 0.45, len(cycle))
    process_temperature = air_temperature + 8.5 + 0.02 * cycle + rng.normal(0, 0.6, len(cycle))
    rotational_speed = 1510.0 + 110 * np.sin(cycle / 6.5) - 95 * asset_bias + rng.normal(0, 38, len(cycle))
    torque = 39.0 + 7.0 * asset_bias + 0.105 * cycle + rng.normal(0, 3.0, len(cycle))
    tool_wear = 10.0 + 1.8 * cycle + 17 * asset_bias + rng.normal(0, 8.5, len(cycle))
    vibration = 1.35 + 0.012 * cycle + 0.62 * asset_bias + rng.normal(0, 0.18, len(cycle))
    pressure = 5.8 - 0.007 * cycle - 0.24 * asset_bias + rng.normal(0, 0.10, len(cycle))
    maintenance_due = ((cycle + asset_numbers * 3) % 28 == 0).astype(int)

    # The simulated failure mode occurs when a repeatable high-load operating
    # condition combines high ambient temperature, speed and vibration.  A
    # deterministic label keeps the bundled happy path reproducible even when
    # XGBoost is unavailable and the sklearn boosting fallback is selected.
    operating_stress = (ambient_temp > 25.0) & (rotational_speed > 1470.0) & (vibration > 1.8)
    failure = operating_stress.astype(int)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": [f"ASSET-{number:03d}" for number in asset_numbers],
            "machine_type": machine_type,
            "ambient_temp_c": ambient_temp.round(3),
            "air_temperature_k": air_temperature.round(3),
            "process_temperature_k": process_temperature.round(3),
            "rotational_speed_rpm": rotational_speed.round(3),
            "torque_nm": torque.round(3),
            "tool_wear_min": np.clip(tool_wear, 0, None).round(3),
            "vibration_mm_s": np.clip(vibration, 0.1, None).round(3),
            "pressure_bar": np.clip(pressure, 0.1, None).round(3),
            "maintenance_due": maintenance_due,
            "failure": failure,
        }
    )

    # Small, deterministic quality imperfections exercise the data-quality UI
    # and imputation pipeline without making the happy path fragile.
    missing_rows = np.arange(17, len(data), 211)
    data.loc[missing_rows, "vibration_mm_s"] = np.nan
    data.loc[np.arange(53, len(data), 307), "machine_type"] = None
    return data.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(drop=True)


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample CSV, creating it deterministically if required."""
    if not SAMPLE_DATA_PATH.exists():
        SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_dataset().to_csv(SAMPLE_DATA_PATH, index=False)
    return pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["timestamp"])
