"""Tests for fraud_detection.evaluate (SPEC Sections 9-10 / QA_PLAN
evaluation acceptance criteria)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_detection.config import Config
from fraud_detection.evaluate import compute_metrics, write_model_card

Y_TRUE = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
Y_PROB = np.array([0.1, 0.2, 0.05, 0.3, 0.9, 0.4, 0.8, 0.15, 0.6, 0.7])


def test_pr_auc_and_roc_auc_match_sklearn_directly() -> None:
    metrics = compute_metrics(Y_TRUE, Y_PROB, target_precision=0.9)
    assert metrics["pr_auc"] == average_precision_score(Y_TRUE, Y_PROB)
    assert metrics["roc_auc"] == roc_auc_score(Y_TRUE, Y_PROB)


def test_recall_at_precision_matches_manual_derivation() -> None:
    target_precision = 0.75
    metrics = compute_metrics(Y_TRUE, Y_PROB, target_precision=target_precision)

    precision, recall, thresholds = precision_recall_curve(Y_TRUE, Y_PROB)
    eligible = precision[:-1] >= target_precision
    assert np.any(eligible)
    expected_recall = recall[:-1][eligible].max()
    best_idx = np.flatnonzero((recall[:-1] == expected_recall) & eligible)[0]
    expected_threshold = thresholds[best_idx]

    assert metrics["recall_at_precision"] == expected_recall
    assert metrics["threshold"] == expected_threshold


def test_unreachable_target_precision_returns_none_without_raising() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # all identical scores
    metrics = compute_metrics(y_true, y_prob, target_precision=0.99)
    assert metrics["threshold"] is None
    assert metrics["recall_at_precision"] is None


def test_confusion_matrix_keys_and_total() -> None:
    metrics = compute_metrics(Y_TRUE, Y_PROB, target_precision=0.9)
    cm = metrics["confusion_matrix"]
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert sum(cm.values()) == len(Y_TRUE)


def test_confusion_matrix_stable_shape_when_threshold_unreachable() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    metrics = compute_metrics(y_true, y_prob, target_precision=0.99)
    cm = metrics["confusion_matrix"]
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert sum(cm.values()) == len(y_true)
    # threshold=None is treated as an always-negative classifier.
    assert cm["tp"] == 0
    assert cm["fp"] == 0


def test_write_model_card_contains_required_content(tmp_path) -> None:
    config = Config(model_type="logistic_regression")
    metrics = compute_metrics(Y_TRUE, Y_PROB, target_precision=0.9)
    path = tmp_path / "model_card.md"

    write_model_card(config, metrics, str(path))

    content = path.read_text(encoding="utf-8").lower()
    assert "synthetic" in content
    assert config.model_type.lower() in content
    assert str(metrics["pr_auc"]).lower() in content
