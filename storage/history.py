"""Best-effort JSON event persistence for a local Streamlit demo.

The authoritative live workflow state remains in ``st.session_state``.  This
store supplies a small durable audit trail when the local filesystem permits
it (as it does in normal local runs); failures are surfaced to callers rather
than breaking a promotion after the user has already approved it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.config import DEFAULT_HISTORY_PATH


def _json_safe(value: Any) -> Any:
    """Convert nested values to a conservative JSON-safe representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class JsonHistoryStore:
    """A compact append-only JSON-list audit store with atomic replacement."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_HISTORY_PATH

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise TypeError("History events must be mappings.")
        events = self.load()
        normalized = _json_safe(dict(event))
        events.insert(0, normalized)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            temporary.write_text(json.dumps(events, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return normalized


def load_events(path: str | Path | None = None) -> list[dict[str, Any]]:
    return JsonHistoryStore(path).load()


def append_event(event: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    return JsonHistoryStore(path).append(event)


__all__ = ["JsonHistoryStore", "append_event", "load_events"]
