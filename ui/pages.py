"""Page renderers for the ReliabilityOS industrial dashboard."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from ui import backend
from ui.components import (
    add_activity,
    empty_panel,
    kpi_card,
    metric_value,
    render_activity_feed,
    section_header,
    status_badge,
)


def initialize_demo_state() -> None:
    """Set non-destructive defaults for all shared demo-state keys."""
    defaults: dict[str, Any] = {
        "profile": None,
        "prepared_data": None,
        "experiments": [],
        "champion": None,
        "critic_report": None,
        "production_data": None,
        "production_metrics": None,
        "drift_report": None,
        "improvement_proposal": None,
        "challenger": None,
        "challenger_critic": None,
        "promotion_history": [],
        "activity_log": [],
        "source_data": None,
        "source_name": None,
        "deployed": False,
        "comparison": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.activity_log:
        add_activity("ReliabilityOS standing by for a predictive-maintenance dataset.", "INFO")


def _run_action(label: str, action: Callable[[], Any]) -> Any | None:
    """Run one backend action with a clear, graceful UI error state."""
    try:
        with st.spinner(label):
            return action()
    except backend.BackendUnavailable as error:
        st.info(f"This capability is loading: {error}")
        add_activity(str(error), "PENDING")
    except Exception as error:  # UI must remain usable after a malformed upload/model.
        st.error(f"The action could not complete: {error}")
        add_activity(f"{label} needs review: {error}", "WARN")
    return None


def _metrics(experiment: dict[str, Any] | None) -> dict[str, Any]:
    return (experiment or {}).get("metrics") or {}


def _experiment_name(experiment: dict[str, Any] | None) -> str:
    if not experiment:
        return "No model selected"
    return str(experiment.get("model_name") or experiment.get("experiment_id") or "Unnamed candidate")


def _leaderboard_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in experiments:
        metrics = _metrics(result)
        try:
            # Score calculation lives with the ML implementation; the UI only
            # displays its transparent, recall-forward result.
            from ml.training import weighted_selection_score

            selection_score = weighted_selection_score(metrics)
        except (ImportError, TypeError, ValueError):
            selection_score = None
        rows.append(
            {
                "Model": _experiment_name(result),
                "Recall": metrics.get("recall"),
                "Precision": metrics.get("precision"),
                "F1": metrics.get("f1"),
                "False alarm rate": metrics.get("false_alarm_rate"),
                "ROC-AUC": metrics.get("roc_auc"),
                "Brier": metrics.get("brier_score"),
                "Latency (ms)": metrics.get("latency_ms"),
                "Selection score": selection_score,
            }
        )
    return rows


def _profile_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        ("Rows", "rows"),
        ("Columns", "columns"),
        ("Target", "target_column"),
        ("Timestamp", "timestamp_column"),
        ("Asset identifier", "asset_id_column"),
        ("Recommended task", "recommended_task"),
        ("Missing data", "missing_pct"),
    )
    # A single display type avoids Arrow coercion warnings when the profile
    # mixes counts, percentages, and column-name strings.
    return [{"Signal": label, "Value": str(profile.get(key, "—"))} for label, key in keys]


def _critic_context(prepared_data: dict[str, Any] | None) -> dict[str, Any]:
    prepared_data = prepared_data or {}
    return {
        "split_metadata": prepared_data.get("split_metadata"),
        "feature_lineage": prepared_data.get("feature_lineage"),
    }


def _run_autonomous_pipeline(source_data: Any | None = None, source_name: str | None = None) -> bool:
    """Execute the demo path through a candidate recommendation, not promotion.

    This is deliberately an orchestration layer: each domain decision remains
    inside the profiler, ML, critic, monitor, and improvement modules.  The
    final promotion stays out of this routine because a human must approve it.
    """
    try:
        with st.spinner("ReliabilityOS is profiling, training, validating, and investigating drift…"):
            data = source_data if source_data is not None else backend.load_sample()
            name = source_name or "Bundled factory telemetry (simulation)"
            add_activity(f"Ingested {len(data):,} rows from {name}.", "PASS")

            profile = backend.profile_data(data)
            prepared = backend.prepare_data(data, profile)
            add_activity(
                f"Detected {profile.get('recommended_task', 'a predictive-maintenance task')} and created a leakage-safe split.",
                "PASS",
            )

            experiments = backend.train_arena(prepared, profile)
            champion = backend.choose_champion(experiments)
            if champion is None:
                raise RuntimeError("The candidate arena did not produce a deployable model.")
            add_activity(f"Selected {_experiment_name(champion)} as the recall-weighted champion.", "PASS")

            critic_report = backend.validate_candidate(champion, profile, _critic_context(prepared))
            if critic_report.get("recommendation") == "REJECT":
                raise RuntimeError("The Reliability Critic rejected the selected champion; deployment is blocked.")
            add_activity(
                f"Reliability Critic: {critic_report.get('recommendation')} for the champion.",
                critic_report.get("status", "INFO"),
            )

            production_data = backend.simulate_drift(data, profile)
            drift_report = backend.detect_drift(data, production_data, profile)
            production_metrics = backend.evaluate_production(champion, production_data, profile)
            add_activity(
                f"Simulated operating-condition drift; monitor status is {drift_report.get('overall_status')}.",
                drift_report.get("overall_status", "INFO"),
            )

            proposal = backend.create_improvement_proposal(drift_report, champion, profile)
            challenger = backend.train_challenger(prepared, proposal, profile)
            comparison = backend.compare_models(champion, challenger)
            challenger_critic = backend.validate_candidate(challenger, profile, _critic_context(prepared))
            add_activity(
                f"Trained and reviewed {_experiment_name(challenger)}; awaiting human promotion decision.",
                challenger_critic.get("status", "INFO"),
            )
    except backend.BackendUnavailable as error:
        st.error(f"A required demo capability is unavailable: {error}")
        add_activity(str(error), "FAIL")
        return False
    except Exception as error:
        st.error(f"Autonomous run stopped safely: {error}")
        add_activity(f"Autonomous run needs review: {error}", "FAIL")
        return False

    st.session_state.update(
        {
            "source_data": data,
            "source_name": name,
            "profile": profile,
            "prepared_data": prepared,
            "experiments": experiments,
            "champion": champion,
            "critic_report": critic_report,
            "deployed": True,
            "production_data": production_data,
            "production_metrics": production_metrics,
            "drift_report": drift_report,
            "improvement_proposal": proposal,
            "challenger": challenger,
            "comparison": comparison,
            "challenger_critic": challenger_critic,
        }
    )
    return True


def _render_autonomous_findings() -> None:
    """Present the useful conclusions of a completed run without page hopping."""
    profile = st.session_state.profile or {}
    champion = st.session_state.champion
    critic = st.session_state.critic_report or {}
    drift = st.session_state.drift_report or {}
    production = st.session_state.production_metrics or {}
    challenger = st.session_state.challenger
    comparison = st.session_state.comparison or {}
    challenger_critic = st.session_state.challenger_critic or {}

    if not champion:
        empty_panel("Run the autonomous demo to generate a model recommendation and its evidence.")
        return

    baseline = _metrics(champion)
    production_metrics = production.get("metrics") or {}
    cards = st.columns(4)
    with cards[0]:
        kpi_card("Dataset", f"{profile.get('rows', 0):,} rows", profile.get("recommended_task", "—"))
    with cards[1]:
        kpi_card("Champion", _experiment_name(champion), critic.get("recommendation", "REVIEW_REQUIRED"))
    with cards[2]:
        kpi_card("Drift", drift.get("overall_status", "NOT RUN"), f"PSI {metric_value(drift.get('score'))}")
    with cards[3]:
        kpi_card("Challenger", _experiment_name(challenger), comparison.get("recommendation", "AWAITING REVIEW"))

    st.markdown("### What ReliabilityOS found")
    top_features = drift.get("features", [])[:3]
    feature_text = ", ".join(
        f"{item.get('feature')} ({item.get('severity', 'LOW').lower()})" for item in top_features
    ) or "no numeric shift evidence"
    recall_before = metric_value(baseline.get("recall"))
    recall_after = metric_value(production_metrics.get("recall")) if production.get("labels_available") else "unlabelled"
    f1_before = metric_value(baseline.get("f1"))
    f1_after = metric_value(production_metrics.get("f1")) if production.get("labels_available") else "unlabelled"
    st.markdown(
        f"""
        - **Task and data:** {profile.get('recommended_task', 'Predictive maintenance')} on {profile.get('rows', 0):,} telemetry rows.
        - **Best initial model:** **{_experiment_name(champion)}**; the independent critic recommends **{critic.get('recommendation', 'REVIEW_REQUIRED')}**.
        - **Operating change:** the simulated monitor found **{drift.get('overall_status', 'no')}** drift, led by {feature_text}.
        - **Observed impact:** failure recall changed from **{recall_before}** to **{recall_after}** and F1 from **{f1_before}** to **{f1_after}** on labelled simulated production telemetry.
        - **Recommended next step:** {comparison.get('reason', 'Inspect challenger evidence before deciding.')}
        """
    )

    with st.expander("Inspect supporting evidence"):
        evidence_left, evidence_right = st.columns(2)
        with evidence_left:
            st.markdown("#### Candidate leaderboard")
            st.dataframe(_leaderboard_rows(st.session_state.experiments), use_container_width=True, hide_index=True)
            st.markdown("#### Champion critic checks")
            _render_critic_report(critic)
        with evidence_right:
            st.markdown("#### Drift evidence")
            _render_drift_report(drift)
            st.markdown("#### Improvement hypothesis")
            st.write((st.session_state.improvement_proposal or {}).get("reason", "No proposal generated."))
            st.markdown("#### Challenger critic checks")
            _render_critic_report(challenger_critic)


def _render_model_and_monitoring_plots() -> None:
    """Show compact evidence plots derived from trained-model and monitor outputs."""
    champion = st.session_state.champion
    if not champion:
        return
    production = st.session_state.production_metrics or {}
    production_metrics = production.get("metrics") or {}
    drift = st.session_state.drift_report or {}

    st.markdown("### Model and monitoring plots")
    with st.expander("Show model quality, feature evidence, and drift plots", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Candidate quality")
            leaderboard = pd.DataFrame(_leaderboard_rows(st.session_state.experiments))
            if not leaderboard.empty:
                st.bar_chart(leaderboard.set_index("Model")[["Recall", "Precision", "F1"]])
                st.caption("Higher is better. Champion selection also penalizes false alarms and poor calibration.")

            st.markdown("#### Champion feature evidence")
            try:
                importance = backend.model_feature_importance(champion, top_n=8)
            except (backend.BackendUnavailable, TypeError, ValueError):
                importance = []
            if importance:
                importance_frame = pd.DataFrame(importance).set_index("feature")
                st.bar_chart(importance_frame[["importance"]])
                st.caption("Native tree importance or absolute linear coefficient; not an LLM-generated explanation.")
            else:
                st.caption("This selected model does not expose native feature-weight evidence.")

        with right:
            st.markdown("#### Champion vs drift-window performance")
            baseline = _metrics(champion)
            if production.get("labels_available") and production_metrics:
                comparison = pd.DataFrame(
                    {
                        "Champion holdout": [baseline.get("recall"), baseline.get("f1"), baseline.get("false_alarm_rate")],
                        "Simulated drift window": [
                            production_metrics.get("recall"),
                            production_metrics.get("f1"),
                            production_metrics.get("false_alarm_rate"),
                        ],
                    },
                    index=["Failure recall", "F1", "False alarm rate"],
                )
                st.bar_chart(comparison)
                st.caption("Lower false-alarm rate is better; the other two metrics are higher-is-better.")
            else:
                st.caption("Labelled simulated production is required to plot performance degradation.")

            st.markdown("#### Sensor-distribution drift")
            drift_features = pd.DataFrame(drift.get("features", [])[:8])
            if not drift_features.empty:
                st.bar_chart(drift_features.set_index("feature")[["score"]])
                st.caption("Population Stability Index (PSI); larger bars indicate more distribution shift.")
            else:
                st.caption("No numeric sensor drift evidence is available yet.")


def _render_human_approval_gate() -> None:
    challenger = st.session_state.challenger
    if not challenger:
        if st.session_state.promotion_history:
            latest = st.session_state.promotion_history[0]
            st.success(f"Human approval recorded: {latest.get('to', 'challenger')} is now the champion.")
        return

    critic = st.session_state.challenger_critic or {}
    is_rejected = critic.get("recommendation") == "REJECT"
    st.markdown("### Human approval required")
    st.caption("The system can investigate and recommend, but it cannot promote a model on its own.")
    approve, rationale = st.columns([1, 2])
    with approve:
        if st.button("Approve challenger promotion", type="primary", disabled=is_rejected, use_container_width=True):
            from core.state import record_human_approved_promotion
            from storage.history import append_event

            entry = record_human_approved_promotion(
                st.session_state,
                st.session_state.champion,
                challenger,
                critic,
                st.session_state.comparison,
            )
            try:
                append_event(entry)
            except OSError:
                add_activity("Promotion is recorded for this session; local audit-file persistence was unavailable.", "WARN")
            st.session_state.champion = challenger
            st.session_state.critic_report = critic or None
            st.session_state.challenger = None
            st.session_state.challenger_critic = None
            st.session_state.comparison = None
            add_activity(f"Human approved promotion to {entry['to']}; experiment knowledge updated.", "APPROVED")
            st.rerun()
    with rationale:
        if is_rejected:
            st.error("Promotion is blocked because the Reliability Critic rejected this challenger.")
        else:
            st.info(
                f"Recommendation: {st.session_state.comparison.get('recommendation', 'REVIEW_REQUIRED') if st.session_state.comparison else 'REVIEW_REQUIRED'}. "
                "Approve only after reviewing the evidence above."
            )


def render_autonomous_demo() -> None:
    """Render the simplified, one-screen ReliabilityOS demo experience."""
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Autonomous predictive-maintenance demo</div>
          <h1>ReliabilityOS</h1>
          <p>One action profiles telemetry, trains and validates a champion, simulates drift,
          recommends a challenger, and leaves the final promotion decision to you.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    control, activity = st.columns([1.35, 1])
    with control:
        st.markdown("### Run the autonomous reliability loop")
        st.caption("Uses bundled factory telemetry and deterministic local ML. No LLM or external service is used.")
        if st.button("Run autonomous demo", type="primary", use_container_width=True):
            _run_autonomous_pipeline()
        with st.expander("Optional: run a CSV instead of the bundled demo data"):
            upload = st.file_uploader("Telemetry CSV", type=["csv"], key="autonomous_upload")
            if st.button("Run uploaded CSV", use_container_width=True, disabled=upload is None):
                data = _run_action("Reading uploaded CSV…", lambda: backend.load_csv(upload))
                if data is not None:
                    _run_autonomous_pipeline(data, upload.name)
    with activity:
        render_activity_feed(compact=False)

    st.divider()
    _render_autonomous_findings()
    _render_model_and_monitoring_plots()
    _render_human_approval_gate()
    st.divider()
    st.caption(
        "Simulation notice: factory telemetry, operating drift, and labelled production impact are deterministic demo simulations. "
        "ReliabilityOS improves the predictive-maintenance pipeline; it does not continuously fine-tune a foundation model."
    )


def render_overview() -> None:
    champion = st.session_state.champion
    drift = st.session_state.drift_report or {}
    profile = st.session_state.profile or {}
    deployed = bool(st.session_state.deployed)
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Predictive maintenance control plane</div>
          <h1>ReliabilityOS</h1>
          <p>Autonomous, evidence-backed improvement for industrial failure prediction.
          Every promotion remains gated by human approval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    cards = st.columns(4)
    with cards[0]:
        kpi_card("Dataset", f"{profile.get('rows', 0):,} rows" if profile else "Not loaded", profile.get("recommended_task", "Awaiting intake"))
    with cards[1]:
        kpi_card("Champion", _experiment_name(champion), "Deployed" if deployed else "Not deployed")
    with cards[2]:
        kpi_card("Production health", drift.get("overall_status", "NOT RUN"), "Simulated operating telemetry")
    with cards[3]:
        kpi_card("Improvement", "Ready" if st.session_state.improvement_proposal else "Standing by", "Human approval required")

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        section_header("Autonomous reliability loop", "The demo moves a dataset through an auditable ML lifecycle.")
        complete = 0
        if st.session_state.source_data is not None:
            complete = 2
        if st.session_state.experiments:
            complete = 4
        if deployed:
            complete = 6
        if st.session_state.drift_report:
            complete = 8
        if st.session_state.challenger:
            complete = 10
        if st.session_state.promotion_history:
            complete = 12
        try:
            import plotly.graph_objects as go

            stages = [
                "Data", "Profile", "Plan", "Train", "Critique", "Deploy", "Monitor",
                "Detect drift", "Improve", "Compare", "Human approve", "Learn",
            ]
            colors = ["#27a99b" if index < complete else "#1b4055" for index in range(len(stages))]
            figure = go.Figure(
                go.Scatter(
                    x=list(range(len(stages))), y=[1] * len(stages), mode="lines+markers+text",
                    line={"color": "#2d6c7c", "width": 3},
                    marker={"size": 17, "color": colors}, text=stages, textposition="bottom center",
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            figure.update_layout(
                height=195, margin={"l": 10, "r": 10, "t": 10, "b": 45},
                xaxis={"visible": False}, yaxis={"visible": False}, paper_bgcolor="#07131f", plot_bgcolor="#07131f",
                font={"color": "#c7d6e0", "size": 11},
            )
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        except ImportError:
            st.progress(min(1.0, complete / 12), text="Workflow progress")
    with right:
        render_activity_feed(compact=True)

    st.write("")
    if st.session_state.source_data is None:
        st.info("Start in **Data Intake**: load the bundled machine-failure dataset and create a profile.")
    elif not champion:
        st.info("Data intelligence is ready. Continue to **AutoML Arena** to train candidate models.")
    elif not deployed:
        st.info("A champion is available. Review it in **Reliability Critic**, then deploy to simulated production.")


def render_data_intake() -> None:
    section_header("Data Intake", "Ingest a local CSV or start with the bundled synthetic factory telemetry.")
    first, second = st.columns([1, 1.35])
    with first:
        st.markdown("#### Demo dataset")
        st.caption("Deterministic synthetic telemetry — clearly labeled simulation, with failure labels and timestamps.")
        if st.button("Load sample factory telemetry", type="primary", use_container_width=True):
            data = _run_action("Loading bundled telemetry…", backend.load_sample)
            if data is not None:
                st.session_state.source_data = data
                st.session_state.source_name = "Bundled factory telemetry (simulation)"
                st.session_state.profile = None
                st.session_state.prepared_data = None
                add_activity(f"Loaded {len(data):,} rows of bundled factory telemetry.", "PASS")
                st.rerun()
    with second:
        st.markdown("#### CSV upload")
        upload = st.file_uploader("Bring a telemetry CSV", type=["csv"], label_visibility="collapsed")
        if upload is not None and st.button("Use uploaded CSV", use_container_width=True):
            data = _run_action("Reading uploaded CSV…", lambda: backend.load_csv(upload))
            if data is not None:
                st.session_state.source_data = data
                st.session_state.source_name = upload.name
                st.session_state.profile = None
                st.session_state.prepared_data = None
                add_activity(f"Ingested uploaded dataset: {upload.name} ({len(data):,} rows).", "PASS")
                st.rerun()

    data = st.session_state.source_data
    if data is None:
        empty_panel("No dataset loaded. Use the sample telemetry to take the complete workflow immediately.")
        return

    st.write("")
    name, controls = st.columns([2.5, 1])
    with name:
        st.markdown(f"#### Active source: `{st.session_state.source_name or 'Dataset'}`")
    with controls:
        if st.button("Profile & prepare", type="primary", use_container_width=True):
            profile = _run_action("Profiling schema and task…", lambda: backend.profile_data(data))
            if profile is not None:
                prepared = _run_action("Preparing leakage-safe features…", lambda: backend.prepare_data(data, profile))
                if prepared is not None:
                    st.session_state.profile = profile
                    st.session_state.prepared_data = prepared
                    warnings = len(profile.get("warnings", []))
                    add_activity(
                        f"Profiled {profile.get('rows', 0):,} rows; recommended {profile.get('recommended_task', 'task')} ({warnings} warnings).",
                        "WARN" if warnings else "PASS",
                    )
                    add_activity("Prepared time-aware feature splits without future-row leakage.", "PASS")
                    st.rerun()

    tab_preview, tab_profile = st.tabs(["Telemetry preview", "Data profile"])
    with tab_preview:
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    with tab_profile:
        profile = st.session_state.profile
        if not profile:
            empty_panel("Run Profile & prepare to inspect the schema, target recommendation, and data-quality evidence.")
        else:
            columns = st.columns(4)
            with columns[0]:
                kpi_card("Rows", f"{profile.get('rows', 0):,}")
            with columns[1]:
                kpi_card("Numeric sensors", len(profile.get("numeric_columns", [])))
            with columns[2]:
                kpi_card("Missing values", f"{float(profile.get('missing_pct', 0)):.1%}")
            with columns[3]:
                kpi_card("Task", profile.get("recommended_task", "—"))
            st.dataframe(_profile_rows(profile), use_container_width=True, hide_index=True)
            warnings = profile.get("warnings", [])
            if warnings:
                st.warning(" · ".join(map(str, warnings)))
            balance = profile.get("class_balance")
            if balance:
                st.caption("Target distribution")
                st.json(balance)


def render_automl_arena() -> None:
    section_header("AutoML Arena", "Bounded candidate training scored for industrial recall, false alarms, and latency.")
    prepared = st.session_state.prepared_data
    if not prepared:
        empty_panel("Profile and prepare a dataset before launching the candidate arena.")
        return
    controls, summary = st.columns([1, 1.7])
    with controls:
        if st.button("Train candidate arena", type="primary", use_container_width=True):
            experiments = _run_action(
                "Training bounded candidate models…",
                lambda: backend.train_arena(prepared, st.session_state.profile),
            )
            if experiments is not None:
                st.session_state.experiments = experiments
                st.session_state.champion = None
                st.session_state.critic_report = None
                add_activity(f"AutoML Arena evaluated {len(experiments)} candidates with industrial metrics.", "PASS")
                st.rerun()
    with summary:
        st.caption("Candidates: Logistic Regression · Random Forest · XGBoost (or sklearn boosting fallback).")
        st.caption("Temporal holdout is honored when the dataset has a timestamp.")

    experiments = st.session_state.experiments
    if not experiments:
        empty_panel("No candidates trained yet. Training is deterministic and sized for interactive Streamlit use.")
        return
    st.write("")
    rows = _leaderboard_rows(experiments)
    st.markdown("#### Candidate leaderboard")
    st.dataframe(rows, use_container_width=True, hide_index=True, column_config={
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
        "False alarm rate": st.column_config.NumberColumn(format="%.3f"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Brier": st.column_config.NumberColumn(format="%.3f"),
        "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
        "Selection score": st.column_config.NumberColumn(format="%.3f"),
    })
    options = {_experiment_name(item): item for item in experiments}
    selected_name = st.selectbox("Candidate to designate", list(options), key="arena_candidate")
    left, right = st.columns([1, 2])
    with left:
        if st.button("Select champion", type="primary", use_container_width=True):
            champion = _run_action("Selecting reliability-weighted champion…", lambda: backend.choose_champion(experiments))
            champion = champion or options.get(selected_name)
            if champion:
                st.session_state.champion = champion
                add_activity(f"Selected {_experiment_name(champion)} as champion candidate.", "PASS")
                st.rerun()
    with right:
        selected = options[selected_name]
        st.caption("Selected model evidence")
        st.json({"model": _experiment_name(selected), "metrics": _metrics(selected), "notes": selected.get("notes", [])})


def _render_critic_report(report: dict[str, Any] | None) -> None:
    if not report:
        empty_panel("No critic report yet. The critic independently checks reliability before deployment.")
        return
    headline, recommendation = st.columns([1, 3])
    with headline:
        st.markdown(status_badge(report.get("status")), unsafe_allow_html=True)
    with recommendation:
        st.markdown(f"**Recommendation:** `{report.get('recommendation', 'REVIEW_REQUIRED')}`")
    checks = report.get("checks", [])
    st.dataframe(
        [{"Check": item.get("name"), "Status": item.get("status"), "Evidence": item.get("evidence")} for item in checks],
        use_container_width=True,
        hide_index=True,
    )


def render_reliability_critic() -> None:
    section_header("Reliability Critic", "Independent deterministic gates for leakage, calibration, recall, false alarms, and latency.")
    champion = st.session_state.champion
    if not champion:
        empty_panel("Select a champion in AutoML Arena before requesting independent validation.")
        return
    name, action = st.columns([2.2, 1])
    with name:
        st.markdown(f"#### Candidate under review: `{_experiment_name(champion)}`")
    with action:
        if st.button("Run critic validation", type="primary", use_container_width=True):
            report = _run_action(
                "Running deterministic Reliability Critic…",
                lambda: backend.validate_candidate(
                    champion,
                    st.session_state.profile,
                    {
                        "split_metadata": (st.session_state.prepared_data or {}).get("split_metadata"),
                        "feature_lineage": (st.session_state.prepared_data or {}).get("feature_lineage"),
                    },
                ),
            )
            if report is not None:
                st.session_state.critic_report = report
                add_activity(
                    f"Critic returned {report.get('status')} / {report.get('recommendation')} for {_experiment_name(champion)}.",
                    report.get("status", "INFO"),
                )
                st.rerun()
    _render_critic_report(st.session_state.critic_report)


def render_production_monitor() -> None:
    status = "DEPLOYED" if st.session_state.deployed else "PENDING"
    section_header("Production Monitor", "Simulated factory telemetry and drift diagnostics — no live PLC integration.", status)
    champion = st.session_state.champion
    report = st.session_state.critic_report or {}
    if not champion:
        empty_panel("A validated champion is required before simulated deployment.")
        return
    recommendation = report.get("recommendation")
    top, deploy_col = st.columns([2.2, 1])
    with top:
        st.markdown(f"#### Champion: `{_experiment_name(champion)}`")
        st.caption(f"Critic disposition: {recommendation or 'not yet validated'}")
    with deploy_col:
        blocked = recommendation == "REJECT"
        if st.button("Deploy to simulation", type="primary", disabled=blocked, use_container_width=True):
            st.session_state.deployed = True
            add_activity(f"Deployed {_experiment_name(champion)} to simulated production monitoring.", "DEPLOYED")
            st.rerun()
        if blocked:
            st.caption("Deployment blocked: critic rejected this candidate.")
    if not st.session_state.deployed:
        empty_panel("Deployment is a demo-safe simulation. Run critic validation and deploy when ready.")
        return
    production_metrics = st.session_state.get("production_metrics") or {}
    has_labelled_production = bool(production_metrics.get("labels_available"))
    metrics = production_metrics.get("metrics") if has_labelled_production else _metrics(champion)
    metrics = metrics or {}
    cols = st.columns(4)
    with cols[0]:
        kpi_card(
            "Failure recall",
            metric_value(metrics.get("recall")),
            "Labelled simulation" if has_labelled_production else "Holdout evaluation",
        )
    with cols[1]:
        kpi_card(
            "False alarms",
            metric_value(metrics.get("false_alarm_rate")),
            "Labelled simulation" if has_labelled_production else "Holdout evaluation",
        )
    with cols[2]:
        kpi_card(
            "Inference latency",
            f"{metric_value(metrics.get('latency_ms'), 2)} ms",
            "Labelled simulation" if has_labelled_production else "Per batch estimate",
        )
    with cols[3]:
        drift = st.session_state.drift_report or {}
        kpi_card("Drift state", drift.get("overall_status", "STABLE"), "Simulated production feed")
    data = st.session_state.production_data
    if data is not None:
        if has_labelled_production:
            st.warning(
                f"Degradation evidence: the metrics above were measured on "
                f"{production_metrics.get('rows_evaluated', 0):,} labelled rows from the simulated drift window."
            )
        st.caption("Latest simulated production telemetry")
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)
    else:
        st.info("Production telemetry remains stable. Use Self-Improvement to inject operating-condition drift.")


def _render_drift_report(report: dict[str, Any] | None) -> None:
    if not report:
        empty_panel("No drift study yet. Simulate an operating condition change to begin diagnosis.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(status_badge(report.get("overall_status")), unsafe_allow_html=True)
    with col2:
        st.metric("Aggregate drift score", metric_value(report.get("score")))
    st.dataframe(
        [
            {"Feature": item.get("feature"), "Drift score": item.get("score"), "Severity": item.get("severity")}
            for item in report.get("features", [])
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_self_improvement() -> None:
    section_header("Self-Improvement", "A deterministic challenger loop. Models never promote without explicit human approval.")
    if not st.session_state.deployed:
        empty_panel("Deploy a champion to simulated production before launching the self-improvement loop.")
        return
    champion = st.session_state.champion
    source = st.session_state.source_data
    profile = st.session_state.profile
    stage_one, stage_two, stage_three = st.columns(3)
    with stage_one:
        st.markdown("#### 1 · Create evidence")
        if st.button("Simulate operating drift", use_container_width=True):
            production = _run_action("Injecting deterministic sensor drift…", lambda: backend.simulate_drift(source, profile))
            if production is not None:
                st.session_state.production_data = production
                st.session_state.production_metrics = None
                st.session_state.drift_report = None
                add_activity("Simulated a shifted operating regime in production telemetry.", "WARN")
                st.rerun()
        if st.button("Detect drift", use_container_width=True, disabled=st.session_state.production_data is None):
            drift = _run_action(
                "Measuring feature drift…",
                lambda: backend.detect_drift(source, st.session_state.production_data, profile),
            )
            if drift is not None:
                st.session_state.drift_report = drift
                production_metrics = _run_action(
                    "Evaluating labelled simulated production telemetry…",
                    lambda: backend.evaluate_production(champion, st.session_state.production_data, profile),
                )
                if production_metrics is not None:
                    st.session_state.production_metrics = production_metrics
                add_activity(f"Drift monitor reported {drift.get('overall_status', 'UNKNOWN')}.", drift.get("overall_status", "INFO"))
                st.rerun()
    with stage_two:
        st.markdown("#### 2 · Improve")
        if st.button("Generate experiment", use_container_width=True, disabled=not st.session_state.drift_report):
            proposal = _run_action(
                "Generating evidence-backed improvement proposal…",
                lambda: backend.create_improvement_proposal(st.session_state.drift_report, champion, profile),
            )
            if proposal is not None:
                st.session_state.improvement_proposal = proposal
                add_activity("Generated a challenger experiment specification from drift evidence.", "PASS")
                st.rerun()
        if st.button("Train challenger", type="primary", use_container_width=True, disabled=not st.session_state.improvement_proposal):
            challenger = _run_action(
                "Training challenger with proposed changes…",
                lambda: backend.train_challenger(st.session_state.prepared_data, st.session_state.improvement_proposal, profile),
            )
            if challenger is not None:
                st.session_state.challenger = challenger
                add_activity(f"Trained challenger {_experiment_name(challenger)}.", "PASS")
                st.rerun()
    with stage_three:
        st.markdown("#### 3 · Govern")
        if st.button("Compare models", use_container_width=True, disabled=not st.session_state.challenger):
            comparison = _run_action(
                "Comparing champion and challenger…",
                lambda: backend.compare_models(champion, st.session_state.challenger),
            )
            if comparison is not None:
                st.session_state.comparison = comparison
                add_activity("Completed champion-versus-challenger comparison.", "PASS")
                st.rerun()
        if st.button("Validate challenger", use_container_width=True, disabled=not st.session_state.challenger):
            report = _run_action(
                "Running Reliability Critic on challenger…",
                lambda: backend.validate_candidate(
                    st.session_state.challenger,
                    profile,
                    {
                        "split_metadata": (st.session_state.prepared_data or {}).get("split_metadata"),
                        "feature_lineage": (st.session_state.prepared_data or {}).get("feature_lineage"),
                    },
                ),
            )
            if report is not None:
                st.session_state.challenger_critic = report
                add_activity(f"Challenger critic result: {report.get('status')}.", report.get("status", "INFO"))
                st.rerun()

    st.write("")
    evidence, proposal_col = st.columns(2)
    with evidence:
        st.markdown("#### Drift evidence")
        _render_drift_report(st.session_state.drift_report)
    with proposal_col:
        st.markdown("#### Improvement experiment")
        proposal = st.session_state.improvement_proposal
        if proposal:
            st.json(proposal)
        else:
            empty_panel("Evidence will determine the proposed feature and model changes.")
    if st.session_state.challenger:
        st.markdown("#### Challenger governance")
        left, right = st.columns(2)
        with left:
            st.caption("Champion vs challenger")
            if st.session_state.comparison:
                st.json(st.session_state.comparison)
            else:
                st.dataframe(_leaderboard_rows([champion, st.session_state.challenger]), use_container_width=True, hide_index=True)
        with right:
            st.caption("Challenger Reliability Critic")
            _render_critic_report(st.session_state.challenger_critic)
        critic = st.session_state.challenger_critic or {}
        is_rejected = critic.get("recommendation") == "REJECT"
        st.divider()
        st.markdown("#### Human approval gate")
        st.caption("Promotion is intentionally manual. This action records the decision in demo history.")
        approval, note = st.columns([1, 2])
        with approval:
            if st.button("Approve & promote", type="primary", disabled=is_rejected, use_container_width=True):
                from core.state import record_human_approved_promotion
                from storage.history import append_event

                entry = record_human_approved_promotion(
                    st.session_state,
                    champion,
                    st.session_state.challenger,
                    critic,
                    st.session_state.comparison,
                )
                try:
                    append_event(entry)
                except OSError:
                    add_activity("Promotion is recorded for this session; local audit-file persistence was unavailable.", "WARN")
                st.session_state.champion = st.session_state.challenger
                st.session_state.critic_report = critic or None
                st.session_state.challenger = None
                st.session_state.challenger_critic = None
                st.session_state.comparison = None
                add_activity(f"Human approved promotion to {entry['to']}; learning history updated.", "APPROVED")
                st.success("Challenger promoted to champion and recorded in experiment history.")
                st.rerun()
        with note:
            if is_rejected:
                st.error("Promotion is blocked because the Reliability Critic rejected the challenger.")
            else:
                st.info("Review the comparison and critic evidence before approving promotion.")


def render_knowledge_graph() -> None:
    section_header("Knowledge Graph", "A lightweight record of assets, models, evidence, and governed promotions.")
    history = st.session_state.promotion_history
    profile = st.session_state.profile or {}
    champion = st.session_state.champion
    try:
        import networkx as nx
        import plotly.graph_objects as go

        from agents.knowledge import build_knowledge_graph

        graph = build_knowledge_graph(
            profile=profile,
            experiments=st.session_state.experiments,
            champion=champion,
            drift_report=st.session_state.drift_report,
            improvement_proposal=st.session_state.improvement_proposal,
            promotion_history=history,
        )
        if not graph.nodes:
            empty_panel("The graph will populate after data intake, training, and a governed promotion.")
            return
        positions = nx.spring_layout(graph, seed=42)
        edges_x, edges_y = [], []
        for start, end in graph.edges:
            x0, y0 = positions[start]
            x1, y1 = positions[end]
            edges_x.extend([x0, x1, None])
            edges_y.extend([y0, y1, None])
        nodes_x, nodes_y = zip(*(positions[node] for node in graph.nodes))
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", line={"color": "#396071", "width": 1.5}, hoverinfo="skip"))
        figure.add_trace(go.Scatter(
            x=nodes_x, y=nodes_y, mode="markers+text", text=list(graph.nodes), textposition="top center",
            marker={"size": 17, "color": "#27a99b", "line": {"color": "#b6ece6", "width": 1}},
            hovertemplate="%{text}<extra></extra>",
        ))
        figure.update_layout(
            height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10}, paper_bgcolor="#07131f", plot_bgcolor="#07131f",
            xaxis={"visible": False}, yaxis={"visible": False}, font={"color": "#dbe8ef"}, showlegend=False,
        )
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    except ImportError:
        st.info("Install the optional visualization dependencies to render the relationship graph.")
    if history:
        st.markdown("#### Promotion history")
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.caption("No promotions recorded. Human approvals will appear here as durable experiment knowledge.")
