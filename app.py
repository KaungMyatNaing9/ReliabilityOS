"""ReliabilityOS — self-improving predictive-maintenance demo."""

from __future__ import annotations

import streamlit as st

from ui.components import inject_styles
from ui.pages import initialize_demo_state, render_autonomous_demo


def main() -> None:
    st.set_page_config(
        page_title="ReliabilityOS | Predictive Maintenance",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    initialize_demo_state()

    render_autonomous_demo()


if __name__ == "__main__":
    main()
