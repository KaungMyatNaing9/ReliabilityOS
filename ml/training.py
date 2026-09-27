"""Bounded, deterministic model training for the ReliabilityOS AutoML arena."""

from __future__ import annotations

from hashlib import sha1
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.evaluation import evaluate_classifier, infer_positive_label, metric_or_default


RANDOM_STATE = 42
MAX_TRAINING_ROWS = 12_000


def _as_frame(features: Any, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a defensive DataFrame while retaining caller column names."""

    if isinstance(features, pd.DataFrame):
        return features.copy()
    if isinstance(features, pd.Series):
        return features.to_frame()
    array = np.asarray(features)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Feature data must be a two-dimensional table.")
    names = list(feature_names or [f"feature_{index}" for index in range(array.shape[1])])
    if len(names) != array.shape[1]:
        names = [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)


def _get_value(prepared_data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = prepared_data.get(key)
        if value is not None:
            return value
    return None


def extract_training_data(
    prepared_data: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], str]:
    """Extract supplied train/evaluation splits without creating a random split.

    ``ml.preprocessing.prepare_dataset`` provides ``X_train``, ``X_val`` and
    ``X_test``.  The aliases keep the public AutoML entry point convenient for
    small standalone demos and tests, but it never silently calls a splitting
    utility: supplied temporal splits remain intact.
    """

    if not isinstance(prepared_data, Mapping):
        raise TypeError("prepared_data must be a mapping containing pre-split features and targets.")
    feature_names = _get_value(prepared_data, "features", "feature_columns")
    x_train = _get_value(prepared_data, "X_train", "train_features")
    y_train = _get_value(prepared_data, "y_train", "train_target")
    # Prefer the held-out test set. Validation is a safe fallback for callers
    # that intentionally provide a two-way temporal split.
    x_eval = _get_value(prepared_data, "X_test", "test_features", "X_val", "val_features")
    y_eval = _get_value(prepared_data, "y_test", "test_target", "y_val", "val_target")
    evaluation_name = "test" if _get_value(prepared_data, "X_test", "test_features") is not None else "validation"
    if x_train is None or y_train is None or x_eval is None or y_eval is None:
        raise ValueError(
            "prepared_data needs X_train, y_train and an evaluation split "
            "(X_test/y_test or X_val/y_val)."
        )
    train = _as_frame(x_train, feature_names)
    evaluation = _as_frame(x_eval, list(train.columns))
    evaluation = evaluation.reindex(columns=train.columns)
    train_target = np.asarray(y_train).reshape(-1)
    evaluation_target = np.asarray(y_eval).reshape(-1)
    if not len(train) or not len(evaluation):
        raise ValueError("Training and evaluation splits must both contain at least one row.")
    if len(train) != len(train_target) or len(evaluation) != len(evaluation_target):
        raise ValueError("Feature and target row counts do not match.")
    return train, train_target, evaluation, evaluation_target, list(train.columns), evaluation_name


def _bounded_sample(
    features: pd.DataFrame, target: np.ndarray, random_state: int
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Downsample very large demos deterministically while retaining each class."""

    if len(features) <= MAX_TRAINING_ROWS:
        return features, target, False

    # A plain uniform sample can (rarely, but disastrously) remove an already
    # scarce failure class.  Proportional allocations preserve class coverage
    # while keeping the hard interactive-demo row bound.
    labels, counts = np.unique(target, return_counts=True)
    raw_allocations = counts / len(target) * MAX_TRAINING_ROWS
    allocations = np.maximum(1, np.floor(raw_allocations).astype(int))
    allocations = np.minimum(allocations, counts)
    remaining = MAX_TRAINING_ROWS - int(allocations.sum())
    remainders = raw_allocations - np.floor(raw_allocations)
    ranked = sorted(range(len(labels)), key=lambda index: (-remainders[index], str(labels[index])))
    while remaining > 0:
        progressed = False
        for index in ranked:
            if allocations[index] < counts[index]:
                allocations[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    rng = np.random.default_rng(random_state)
    chosen_parts = [
        rng.choice(np.flatnonzero(target == label), size=int(size), replace=False)
        for label, size in zip(labels, allocations)
    ]
    chosen = np.sort(np.concatenate(chosen_parts))
    return features.iloc[chosen].copy(), target[chosen], True


def _one_hot_encoder() -> OneHotEncoder:
    """Construct a version-compatible dense encoder for boosting fallback."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a leakage-safe preprocessing transformer fitted inside a pipeline."""

    numeric = list(features.select_dtypes(include=[np.number]).columns)
    categorical = [column for column in features.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were supplied for training.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _class_weight_scale(target: np.ndarray, positive_label: Any) -> float:
    positive = int((target == positive_label).sum())
    negative = len(target) - positive
    if not positive or not negative:
        return 1.0
    return float(max(1.0, negative / positive))


def _sample_weights(target: np.ndarray, positive_label: Any) -> np.ndarray:
    scale = _class_weight_scale(target, positive_label)
    return np.where(target == positive_label, scale, 1.0).astype(float)


def _candidate_definitions(
    random_state: int, target: np.ndarray, positive_label: Any
) -> list[tuple[str, str, Any, bool]]:
    """Return model factories; bool indicates whether fit sample weights are needed."""

    candidates: list[tuple[str, str, Any, bool]] = [
        (
            "Logistic Regression",
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced", max_iter=350, solver="liblinear", random_state=random_state
            ),
            False,
        ),
        (
            "Random Forest",
            "random_forest",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            ),
            False,
        ),
    ]
    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "XGBoost",
                "xgboost",
                XGBClassifier(
                    n_estimators=90,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=_class_weight_scale(target, positive_label),
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                    verbosity=0,
                ),
                False,
            )
        )
    except Exception:  # optional package can also fail to load native bindings
        candidates.append(
            (
                "Gradient Boosting (XGBoost fallback)",
                "gradient_boosting_fallback",
                GradientBoostingClassifier(
                    n_estimators=90,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=random_state,
                ),
                True,
            )
        )
    return candidates


def _candidate_overrides(model_key: str, candidate_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return one model's requested challenger overrides, if any."""

    if not candidate_config:
        return {}
    overrides = candidate_config.get(model_key)
    if overrides is None:
        overrides = candidate_config.get(model_key.replace("_", " "))
    return overrides if isinstance(overrides, Mapping) else {}


def _apply_overrides(model: Any, model_key: str, candidate_config: Mapping[str, Any] | None) -> Any:
    """Apply safe, explicit per-model parameter overrides for challengers."""

    overrides = _candidate_overrides(model_key, candidate_config)
    valid = model.get_params(deep=False)
    allowed = {key: value for key, value in overrides.items() if key in valid}
    return model.set_params(**allowed) if allowed else model


def weighted_selection_score(metrics: Mapping[str, float | None]) -> float:
    """Explainable champion score, prioritising caught failures over alerts.

    Recall carries 50% of the positive score.  Precision, F1 and discrimination
    add supporting evidence, while false alarms receive a substantial direct
    penalty and poorly calibrated probabilities a smaller penalty.
    """

    recall = metric_or_default(metrics, "recall", 0.0)
    precision = metric_or_default(metrics, "precision", 0.0)
    f1 = metric_or_default(metrics, "f1", 0.0)
    roc_auc = metric_or_default(metrics, "roc_auc", 0.5)
    false_alarm_rate = metric_or_default(metrics, "false_alarm_rate", 1.0)
    brier_score = metric_or_default(metrics, "brier_score", 0.5)
    return float(
        0.50 * recall
        + 0.15 * precision
        + 0.15 * f1
        + 0.10 * roc_auc
        - 0.25 * false_alarm_rate
        - 0.05 * min(max(brier_score, 0.0), 1.0)
    )


def _experiment_id(
    model_key: str,
    feature_names: Sequence[str],
    training_rows: int,
    random_state: int,
    candidate_config: Mapping[str, Any] | None = None,
) -> str:
    """Create a stable ID that differentiates parameterised challengers."""

    overrides = _candidate_overrides(model_key, candidate_config)
    config_signature = repr(sorted((str(key), repr(value)) for key, value in overrides.items()))
    fingerprint = "|".join(
        [model_key, str(training_rows), str(random_state), *(str(name) for name in feature_names), config_signature]
    )
    return f"exp-{model_key}-{sha1(fingerprint.encode()).hexdigest()[:8]}"


def train_automl(
    prepared_data: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    *,
    random_state: int = RANDOM_STATE,
    candidate_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train the bounded ReliabilityOS candidate arena on supplied splits.

    The return value is a list of ``ExperimentResult`` dictionaries matching
    ``INTERFACES.md``.  Candidate-level errors are represented as results with
    null metrics, allowing the Streamlit arena to remain informative instead
    of failing an entire demo run.
    """

    del profile  # profile remains accepted public API context; splits are authoritative.
    train, target, evaluation, evaluation_target, feature_names, evaluation_name = extract_training_data(prepared_data)
    unique = np.unique(target)
    if len(unique) < 2:
        raise ValueError("AutoML requires at least two target classes in the training split.")
    if len(unique) > 2:
        raise ValueError("ReliabilityOS demo AutoML currently supports binary classification targets.")
    positive_label = infer_positive_label(target)
    train, target, was_bounded = _bounded_sample(train, target, random_state)
    preprocessor = make_preprocessor(train)
    results: list[dict[str, Any]] = []

    for model_name, model_key, base_model, needs_sample_weights in _candidate_definitions(
        random_state, target, positive_label
    ):
        model = _apply_overrides(base_model, model_key, candidate_config)
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])
        notes = [f"Evaluated on supplied {evaluation_name} split; no random split was created."]
        if was_bounded:
            notes.append(f"Training deterministically capped at {MAX_TRAINING_ROWS:,} rows for demo responsiveness.")
        if model_key == "gradient_boosting_fallback":
            notes.append("XGBoost unavailable; used sklearn GradientBoosting fallback.")
        try:
            fit_params: dict[str, Any] = {}
            if needs_sample_weights:
                fit_params["model__sample_weight"] = _sample_weights(target, positive_label)
            pipeline.fit(train, target, **fit_params)
            metrics = evaluate_classifier(pipeline, evaluation, evaluation_target, positive_label=positive_label)
            score = weighted_selection_score(metrics)
            notes.append(f"Weighted reliability selection score: {score:.3f} (recall-forward, false-alarm-penalized).")
            results.append(
                {
                    "experiment_id": _experiment_id(
                        model_key, feature_names, len(train), random_state, candidate_config
                    ),
                    "model_name": model_name,
                    "model": pipeline,
                    "metrics": metrics,
                    "features": feature_names,
                    "notes": notes,
                }
            )
        except Exception as error:
            notes.append(f"Candidate training failed safely: {error}")
            results.append(
                {
                    "experiment_id": _experiment_id(
                        model_key, feature_names, len(train), random_state, candidate_config
                    ),
                    "model_name": model_name,
                    "model": None,
                    "metrics": {
                        "recall": None,
                        "precision": None,
                        "f1": None,
                        "false_alarm_rate": None,
                        "roc_auc": None,
                        "brier_score": None,
                        "latency_ms": None,
                    },
                    "features": feature_names,
                    "notes": notes,
                }
            )
    return results


def select_champion(experiments: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Choose the best valid experiment using the documented reliability score."""

    eligible = [
        experiment
        for experiment in experiments
        if experiment.get("model") is not None and isinstance(experiment.get("metrics"), Mapping)
    ]
    if not eligible:
        return None
    # Stable name/id ordering makes ties reproducible across runs.
    ranked = sorted(
        eligible,
        key=lambda experiment: (
            -weighted_selection_score(experiment["metrics"]),
            -metric_or_default(experiment["metrics"], "recall", 0.0),
            metric_or_default(experiment["metrics"], "false_alarm_rate", 1.0),
            str(experiment.get("model_name", "")),
            str(experiment.get("experiment_id", "")),
        ),
    )
    return dict(ranked[0])
