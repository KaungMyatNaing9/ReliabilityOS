"""Small, deployment-safe configuration values for the demo."""

from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "ReliabilityOS"
APP_TAGLINE = "Self-improving predictive-maintenance control plane"
DEMO_MODE_LABEL = "Demo mode · deterministic ML · simulated factory telemetry"
DEFAULT_HISTORY_PATH = Path(os.environ.get("RELIABILITYOS_HISTORY_PATH", ".reliabilityos/history.json"))

