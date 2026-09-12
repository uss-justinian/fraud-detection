"""Orchestration (SPEC Section 12).

All file I/O -- reading/writing CSVs, ``joblib.dump``/``load``, creating
``data/``/``models/`` directories -- lives here, not in
``data.py``/``model.py``/``features.py``/``evaluate.py``.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data import generate_transactions
from fraud_detection.evaluate import compute_metrics, write_model_card
from fraud_detection.features import add_features
from fraud_detection.model import build_pipeline


def time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered (non-random) train/test split (SPEC Section 6).

    Sorts by ``timestamp`` ascending, then takes the first
    ``1 - test_size`` fraction as train and the remaining, chronologically
    later, fraction as test.
    """
    ordered = df.sort_values("timestamp", ascending=True, kind="mergesort").reset_index(
        drop=True
    )
    n_train = round(len(ordered) * (1 - test_size))
    train = ordered.iloc[:n_train].reset_index(drop=True)
    test = ordered.iloc[n_train:].reset_index(drop=True)
    return train, test


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _feature_columns(config: Config) -> list[str]:
    return [*config.numeric_features, *config.categorical_features]


def _load_features(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    raw = pd.read_csv(data_path, parse_dates=["timestamp"])
    return add_features(raw)


def run_generate_data(config: Config, output_path: str | None = None) -> str:
    """Generate synthetic transaction data and write it to CSV.

    Returns the path the data was written to.
    """
    path = output_path if output_path is not None else config.data_path
    df = generate_transactions(config)
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return path


def run_train(
    config: Config,
    data_path: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Train, save, and evaluate a model; write its model card.

    Reads the CSV at ``data_path`` (or ``config.data_path``), applies
    ``add_features``, time-splits, builds and fits the pipeline chosen by
    ``config.model_type``, saves it via ``joblib.dump``, evaluates it on
    the held-out test partition, and writes the model card.
    """
    data_path = data_path if data_path is not None else config.data_path
    model_path = model_path if model_path is not None else config.model_path

    featured = _load_features(data_path)
    train_df, test_df = time_split(featured, config.test_size)

    feature_cols = _feature_columns(config)
    pipeline = build_pipeline(config)
    pipeline.fit(train_df[feature_cols], train_df[config.target])

    _ensure_parent_dir(model_path)
    joblib.dump(pipeline, model_path)

    y_prob = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    metrics = compute_metrics(
        test_df[config.target].to_numpy(), y_prob, config.target_precision
    )

    _ensure_parent_dir(config.model_card_path)
    write_model_card(config, metrics, config.model_card_path)

    return metrics


def run_evaluate(
    config: Config,
    data_path: str | None = None,
    model_path: str | None = None,
    target_precision: float | None = None,
) -> dict[str, Any]:
    """Evaluate a saved model on the held-out test partition (never retrains).

    Loads the model at ``model_path`` (or ``config.model_path``) and the CSV
    at ``data_path`` (or ``config.data_path``), applies the same
    ``add_features`` + ``time_split`` logic used in training, and evaluates
    on the test partition only.
    """
    data_path = data_path if data_path is not None else config.data_path
    model_path = model_path if model_path is not None else config.model_path
    precision = target_precision if target_precision is not None else config.target_precision

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)

    featured = _load_features(data_path)
    _, test_df = time_split(featured, config.test_size)

    feature_cols = _feature_columns(config)
    y_prob = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    return compute_metrics(test_df[config.target].to_numpy(), y_prob, precision)
