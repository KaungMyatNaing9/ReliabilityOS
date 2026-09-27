"""Small, dependency-light Streamlit UI components.

The application pages use these helpers to keep presentation code consistent
and deliberately keep ML and workflow logic out of the UI layer.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any, Iterable

import streamlit as st


STATUS_COLORS = {
    "PASS": "#35c78a",
    "APPROVED": "#35c78a",
    "STABLE": "#35c78a",
    "DEPLOYED": "#35c78a",
    "WARN": "#f4ba43",
    "WARNING": "#f4ba43",
    "REVIEW_REQUIRED": "#f4ba43",
    "FAIL": "#f06666",
    "REJECT": "#f06666",
    "DRIFT": "#f06666",
    "PENDING": "#92a3b8",
    "NOT_RUN": "#92a3b8",
}


def inject_styles() -> None:
    """Apply the compact industrial dashboard visual system."""
    st.markdown(
        """
        <style>
        .stApp { background: #07131f; color: #e8f0f7; }
        [data-testid="stSidebar"] { background: #0a1b2a; border-right: 1px solid #18334b; }
        [data-testid="stSidebar"] * { color: #d9e6f0; }
        h1, h2, h3 { letter-spacing: -0.025em; color: #f0f6fa; }
        .hero { padding: 1.2rem 1.4rem; border: 1px solid #1a4156; border-radius: 16px;
                background: radial-gradient(circle at 85% 0%, #123d52 0, #0a1d2d 46%, #081721 100%); }
        .eyebrow { color: #59d6c7; font-size: .76rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .11em; margin-bottom: .45rem; }
        .hero h1 { margin: 0; font-size: 2rem; }
        .hero p { color: #aabccc; margin: .5rem 0 0; max-width: 850px; }
        .kpi-card { padding: .95rem 1rem; background: #0c2030; border: 1px solid #1b3a4e;
                    border-radius: 12px; min-height: 90px; }
        .kpi-label { color: #90a8bb; font-size: .76rem; text-transform: uppercase; letter-spacing: .07em; }
        .kpi-value { color: #f1f7fb; font-size: 1.5rem; font-weight: 700; margin-top: .24rem; }
        .kpi-note { color: #77b8ca; font-size: .76rem; margin-top: .2rem; }
        .status-badge { display: inline-block; border-radius: 999px; padding: .22rem .55rem;
                       color: #061017 !important; font-size: .70rem; font-weight: 800; letter-spacing: .04em; }
        .feed-event { padding: .55rem .1rem; border-bottom: 1px solid #173247; font-size: .83rem; }
        .feed-time { color: #7191a5; font-family: monospace; font-size: .7rem; margin-right: .38rem; }
        .empty-panel { padding: 1.15rem; color: #9db3c3; background: #0b1d2b; border: 1px dashed #2c5268;
                        border-radius: 10px; }
        [data-testid="stMetric"] { background: #0c2030; border: 1px solid #1b3a4e; padding: .7rem; border-radius: 10px; }
        .stButton > button { border-radius: 8px; border-color: #2b7b8a; }
        .stButton > button[kind="primary"] { background: #27a99b; color: #041318; border: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str | None) -> str:
    """Return a safely escaped colored status pill for use in markdown."""
    normalized = str(status or "PENDING").upper()
    color = STATUS_COLORS.get(normalized, "#92a3b8")
    return (
        f'<span class="status-badge" style="background:{color}">'
        f"{escape(normalized.replace('_', ' '))}</span>"
    )


def section_header(title: str, subtitle: str, status: str | None = None) -> None:
    right = f"&nbsp;&nbsp;{status_badge(status)}" if status else ""
    st.markdown(f"## {escape(title)} {right}", unsafe_allow_html=True)
    st.caption(subtitle)


def kpi_card(label: str, value: Any, note: str = "") -> None:
    st.markdown(
        "<div class='kpi-card'>"
        f"<div class='kpi-label'>{escape(str(label))}</div>"
        f"<div class='kpi-value'>{escape(str(value))}</div>"
        f"<div class='kpi-note'>{escape(str(note))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def empty_panel(message: str) -> None:
    st.markdown(f"<div class='empty-panel'>{escape(message)}</div>", unsafe_allow_html=True)


def add_activity(message: str, level: str = "INFO") -> None:
    """Append a bounded, presentation-ready event to the persistent feed."""
    log = st.session_state.setdefault("activity_log", [])
    log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": str(message),
            "level": str(level).upper(),
        },
    )
    del log[60:]


def render_activity_feed(events: Iterable[dict[str, Any]] | None = None, *, compact: bool = False) -> None:
    """Render the cross-page Autonomous Activity event feed."""
    events = list(events if events is not None else st.session_state.get("activity_log", []))
    st.markdown("#### Autonomous Activity")
    st.caption("Agent actions and deterministic workflow decisions")
    if not events:
        st.caption("Waiting for a dataset or workflow action.")
        return
    limit = 7 if compact else 16
    for event in events[:limit]:
        level = str(event.get("level", "INFO")).upper()
        color = STATUS_COLORS.get(level, "#55b5cf")
        timestamp = escape(str(event.get("time", "now")))
        message = escape(str(event.get("message", "")))
        st.markdown(
            "<div class='feed-event'>"
            f"<span class='feed-time'>{timestamp}</span>"
            f"<span style='color:{color}; font-weight:700'>{level}</span> {message}</div>",
            unsafe_allow_html=True,
        )


def metric_value(value: Any, digits: int = 3, fallback: str = "—") -> str:
    """Format optional numeric values consistently without raising in UI."""
    if value is None:
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
