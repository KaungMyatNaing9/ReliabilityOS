import numpy as np

from ml.evaluation import calculate_false_alarm_rate, evaluate_predictions
from ml.training import select_champion, weighted_selection_score


def _experiment(name, recall, false_alarm_rate, precision=0.5):
    return {
        "experiment_id": name.lower().replace(" ", "-"),
        "model_name": name,
        "model": object(),
        "metrics": {
            "recall": recall,
            "precision": precision,
            "f1": (recall + precision) / 2,
            "false_alarm_rate": false_alarm_rate,
            "roc_auc": 0.7,
            "brier_score": 0.15,
            "latency_ms": 2.0,
        },
        "features": ["sensor"],
        "notes": [],
    }


def test_false_alarm_rate_is_false_positives_over_actual_normals():
    # One alert is a false positive among three actual normal operating rows.
    rate = calculate_false_alarm_rate([0, 0, 0, 1], [1, 0, 0, 1])
    assert rate == 1 / 3


def test_evaluation_uses_none_when_false_alarm_rate_is_undefined():
    metrics = evaluate_predictions(np.array([1, 1]), np.array([1, 1]), [0.8, 0.9])
    assert metrics["false_alarm_rate"] is None
    assert metrics["roc_auc"] is None


def test_champion_selection_emphasises_recall_and_penalises_false_alarms():
    alert_heavy = _experiment("Alert-heavy model", recall=0.90, false_alarm_rate=0.85, precision=0.2)
    reliable = _experiment("Reliable model", recall=0.86, false_alarm_rate=0.08, precision=0.65)

    assert weighted_selection_score(reliable["metrics"]) > weighted_selection_score(alert_heavy["metrics"])
    assert select_champion([alert_heavy, reliable])["model_name"] == "Reliable model"
