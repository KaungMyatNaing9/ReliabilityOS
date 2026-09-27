"""Integration acceptance test for the deterministic ReliabilityOS demo loop."""

from agents.critic import evaluate_candidate
from agents.improvement import compare_models, generate_improvement_proposal, train_challenger
from agents.knowledge import build_knowledge_graph
from agents.ml_engineer import select_champion, train_automl
from agents.monitor import detect_drift, evaluate_production_performance, simulate_drift
from agents.profiler import load_sample_data, prepare_dataset, profile_dataset
from core.state import build_promotion_record
from core.workflow import promotion_gate
from storage.history import JsonHistoryStore


def test_bundled_demo_happy_path_is_end_to_end_and_human_gated(tmp_path):
    source = load_sample_data()
    profile = profile_dataset(source)
    prepared = prepare_dataset(source, profile)

    experiments = train_automl(prepared, profile)
    champion = select_champion(experiments)
    assert len(experiments) >= 3
    assert champion is not None

    critic = evaluate_candidate(champion, profile, {"split_metadata": prepared["split_metadata"]})
    assert critic["recommendation"] != "REJECT"

    production = simulate_drift(source, profile)
    drift = detect_drift(source, production, profile)
    production_performance = evaluate_production_performance(champion, production, profile)
    assert drift["overall_status"] == "DRIFT"
    assert production_performance["labels_available"] is True
    assert production_performance["metrics"]["f1"] < champion["metrics"]["f1"]

    proposal = generate_improvement_proposal(drift, champion, profile)
    challenger = train_challenger(prepared, proposal, profile)
    comparison = compare_models(champion, challenger)
    challenger_critic = evaluate_candidate(challenger, profile, {"split_metadata": prepared["split_metadata"]})
    assert comparison["human_approval_required"] is True
    assert challenger_critic["recommendation"] != "REJECT"

    pending = promotion_gate(champion, challenger, challenger_critic)
    assert pending["promoted"] is False
    assert pending["status"] == "PENDING_HUMAN_APPROVAL"
    promoted = promotion_gate(champion, challenger, challenger_critic, human_approved=True)
    assert promoted["promoted"] is True
    assert promoted["champion"]["experiment_id"] == challenger["experiment_id"]

    record = build_promotion_record(champion, challenger, challenger_critic, comparison)
    store = JsonHistoryStore(tmp_path / "history.json")
    store.append(record)
    assert store.load()[0]["event_type"] == "human_approved_promotion"

    graph = build_knowledge_graph(
        profile=profile,
        experiments=experiments,
        champion=promoted["champion"],
        drift_report=drift,
        improvement_proposal=proposal,
        promotion_history=[record],
    )
    assert len(graph.nodes) > 5
    assert any(attributes.get("kind") == "promotion" for _, attributes in graph.nodes(data=True))
