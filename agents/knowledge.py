"""Lightweight relationship graph for ReliabilityOS experiment knowledge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


def _name(experiment: Mapping[str, Any] | None, fallback: str) -> str:
    if not experiment:
        return fallback
    return str(experiment.get("model_name") or experiment.get("experiment_id") or fallback)


def build_knowledge_graph(
    *,
    profile: Mapping[str, Any] | None = None,
    experiments: Sequence[Mapping[str, Any]] | None = None,
    champion: Mapping[str, Any] | None = None,
    drift_report: Mapping[str, Any] | None = None,
    improvement_proposal: Mapping[str, Any] | None = None,
    promotion_history: Sequence[Mapping[str, Any]] | None = None,
) -> nx.DiGraph:
    """Build an inspectable graph from the current demo state.

    It intentionally stores descriptive metadata rather than model objects, so
    the graph can be rendered and serialized safely in a lightweight demo.
    """
    graph = nx.DiGraph(name="ReliabilityOS knowledge graph")
    profile = profile or {}
    asset_name = str(profile.get("asset_id_column") or "Factory asset fleet")
    dataset_name = "Predictive-maintenance telemetry"
    graph.add_node(dataset_name, kind="dataset")
    graph.add_node(asset_name, kind="asset")
    graph.add_edge(dataset_name, asset_name, relation="describes")

    target = profile.get("target_column")
    if target:
        failure_mode = f"Failure signal: {target}"
        graph.add_node(failure_mode, kind="failure_mode")
        graph.add_edge(dataset_name, failure_mode, relation="labels")

    seen_models: set[str] = set()
    for experiment in experiments or []:
        name = _name(experiment, "Unnamed experiment")
        seen_models.add(name)
        graph.add_node(name, kind="model", experiment_id=experiment.get("experiment_id"))
        graph.add_edge(dataset_name, name, relation="trained_on")
        for feature in list(experiment.get("features") or [])[:12]:
            feature_name = f"Feature: {feature}"
            graph.add_node(feature_name, kind="feature")
            graph.add_edge(feature_name, name, relation="feeds")
    if champion:
        champion_name = _name(champion, "Champion")
        if champion_name not in seen_models:
            graph.add_node(champion_name, kind="model", experiment_id=champion.get("experiment_id"))
        graph.nodes[champion_name]["champion"] = True
        graph.add_edge(champion_name, asset_name, relation="serves")

    for item in (drift_report or {}).get("features", [])[:5]:
        if not isinstance(item, Mapping):
            continue
        feature = str(item.get("feature") or "unknown sensor")
        drift_name = f"Drift: {feature}"
        graph.add_node(drift_name, kind="drift", severity=item.get("severity"), score=item.get("score"))
        graph.add_edge(f"Feature: {feature}", drift_name, relation="shifted")

    proposal = improvement_proposal or {}
    if proposal:
        proposal_name = "Improvement experiment"
        graph.add_node(proposal_name, kind="improvement_proposal", reason=proposal.get("reason"))
        for change in proposal.get("feature_changes", [])[:5]:
            graph.add_edge(proposal_name, str(change), relation="proposes")
        for node, attributes in list(graph.nodes(data=True)):
            if attributes.get("kind") == "drift":
                graph.add_edge(node, proposal_name, relation="triggers")

    for index, record in enumerate(promotion_history or [], start=1):
        if not isinstance(record, Mapping):
            continue
        decision = f"Human approval {index}"
        graph.add_node(decision, kind="promotion", recorded_at=record.get("recorded_at"))
        old_name = str(record.get("from") or "Previous champion")
        new_name = str(record.get("to") or "Promoted challenger")
        graph.add_node(old_name, kind="model")
        graph.add_node(new_name, kind="model", champion=True)
        graph.add_edge(old_name, decision, relation="reviewed_for")
        graph.add_edge(decision, new_name, relation="promoted")
    return graph


def graph_payload(graph: nx.DiGraph) -> dict[str, list[dict[str, Any]]]:
    """Return plain node/edge data suitable for any renderer."""
    return {
        "nodes": [{"id": str(node), **dict(attributes)} for node, attributes in graph.nodes(data=True)],
        "edges": [
            {"source": str(source), "target": str(target), **dict(attributes)}
            for source, target, attributes in graph.edges(data=True)
        ],
    }


__all__ = ["build_knowledge_graph", "graph_payload"]
