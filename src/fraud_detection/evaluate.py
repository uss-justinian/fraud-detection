"""Fraud-appropriate evaluation and the model-card artifact (SPEC Sections
9-10).

No I/O except the model card write in ``write_model_card``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_detection.config import Config


def compute_metrics(
    y_true: ArrayLike, y_prob: ArrayLike, target_precision: float
) -> dict[str, Any]:
    """Compute fraud-appropriate evaluation metrics.

    ``pr_auc`` (average precision) is the primary metric; ``roc_auc`` is
    reported for reference only. ``threshold``/``recall_at_precision`` is
    the highest recall among thresholds whose precision is >=
    ``target_precision``, or ``(None, None)`` if no such threshold exists —
    a valid outcome, not an error. ``confusion_matrix`` is always present,
    computed at ``threshold`` (or effectively 1.0 -- always-negative -- when
    no threshold reaches the target precision), so the return shape is
    stable regardless of whether the target precision is reachable.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    threshold, recall_at_precision = _best_recall_at_precision(
        y_true_arr, y_prob_arr, target_precision
    )
    confusion_matrix = _confusion_matrix_at_threshold(y_true_arr, y_prob_arr, threshold)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "target_precision": target_precision,
        "threshold": threshold,
        "recall_at_precision": recall_at_precision,
        "confusion_matrix": confusion_matrix,
    }


def _best_recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float
) -> tuple[float | None, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision/recall carry one extra trailing point (precision=1,
    # recall=0) representing the unreachable "threshold=+inf" cut, which has
    # no entry in `thresholds` -- exclude it when searching real thresholds.
    achievable_precision = precision[:-1]
    achievable_recall = recall[:-1]

    eligible = achievable_precision >= target_precision
    if not np.any(eligible):
        return None, None

    eligible_indices = np.flatnonzero(eligible)
    best_pos = int(np.argmax(achievable_recall[eligible_indices]))
    best_idx = eligible_indices[best_pos]
    return float(thresholds[best_idx]), float(achievable_recall[best_idx])


def _confusion_matrix_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None
) -> dict[str, int]:
    # threshold=None (target precision unreachable) is treated as an
    # effective threshold of 1.0, i.e. an always-negative classifier, so the
    # confusion matrix's shape never depends on whether the target
    # precision was reachable.
    effective_threshold = threshold if threshold is not None else 1.0
    y_pred = y_prob >= effective_threshold
    y_true_bool = y_true.astype(bool)

    tp = int(np.sum(y_pred & y_true_bool))
    fp = int(np.sum(y_pred & ~y_true_bool))
    fn = int(np.sum(~y_pred & y_true_bool))
    tn = int(np.sum(~y_pred & ~y_true_bool))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def write_model_card(config: Config, metrics: dict[str, Any], path: str) -> None:
    """Write a markdown model card to ``path`` (normally
    ``config.model_card_path``).

    Documents data provenance (synthetic, not a real-world performance
    claim), a config snapshot, the computed metrics, and intended use.
    """
    lines = [
        "# Fraud Detection Model Card",
        "",
        "## Data Provenance",
        "",
        (
            "**All data used to train and evaluate this model is synthetic.** "
            "It was produced by `fraud_detection.data.generate_transactions` "
            "and is not fit to, sampled from, or validated against any real "
            "payments dataset. No real transaction data, PII, or regulated "
            "data of any kind was used."
        ),
        "",
        (
            "**Metrics reported below are not a claim of real-world "
            "fraud-detection performance** and must not be presented as such."
        ),
        "",
        "## Configuration",
        "",
        f"- `random_seed`: {config.random_seed}",
        f"- `n_samples`: {config.n_samples}",
        f"- `fraud_rate`: {config.fraud_rate}",
        f"- `test_size`: {config.test_size}",
        f"- `model_type`: {config.model_type}",
        "",
        "## Metrics",
        "",
        f"- `pr_auc` (primary metric): {metrics['pr_auc']}",
        f"- `roc_auc` (secondary, reported for reference only): {metrics['roc_auc']}",
        f"- `target_precision`: {metrics['target_precision']}",
        f"- `recall_at_precision`: {metrics['recall_at_precision']}",
        f"- `threshold`: {metrics['threshold']}",
        f"- `confusion_matrix`: {metrics['confusion_matrix']}",
        "",
        "## Intended Use",
        "",
        (
            "This model and pipeline are for **educational/portfolio "
            "demonstration** of fraud-detection ML pipeline engineering "
            "practices only. It is **not** intended for production fraud "
            "decisioning, credit decisioning, or any use affecting real "
            "customers."
        ),
        "",
        "## Limitations",
        "",
        (
            "- Fraud patterns are synthetically constructed and simplified; "
            "they do not capture the full complexity or adversarial "
            "adaptation of real-world fraud."
        ),
        (
            '- Class imbalance is handled via `class_weight="balanced"` only; '
            "no resampling technique (synthetic minority oversampling, "
            "random oversampling, or undersampling) is applied."
        ),
        (
            "- The train/test split is time-ordered, not randomly shuffled, "
            "to avoid leaking future transaction patterns into training."
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
