from agents.ml_engineer import select_champion, train_automl
from agents.profiler import load_sample_data, prepare_dataset, profile_dataset
from ml.explain import model_feature_importance


def test_feature_importance_returns_native_evidence_for_trained_champion():
    data = load_sample_data()
    profile = profile_dataset(data)
    champion = select_champion(train_automl(prepare_dataset(data, profile), profile))

    evidence = model_feature_importance(champion, top_n=5)

    assert champion is not None
    assert 1 <= len(evidence) <= 5
    assert all(item["importance"] >= 0 for item in evidence)
