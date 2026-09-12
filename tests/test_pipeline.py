"""Tests for fraud_detection.pipeline (SPEC Sections 6 & 12 / QA_PLAN split
and CLI/pipeline integration acceptance criteria)."""

from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
import pytest

from fraud_detection.config import Config
from fraud_detection.data import generate_transactions
from fraud_detection.features import add_features
from fraud_detection.pipeline import (
    run_evaluate,
    run_generate_data,
    run_train,
    time_split,
)


def _featured_frame(n_samples: int = 300, seed: int = 11) -> pd.DataFrame:
    config = Config(n_samples=n_samples, random_seed=seed)
    return add_features(generate_transactions(config))


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.3])
def test_time_split_sizes_and_ordering(test_size: float) -> None:
    df = _featured_frame()
    train, test = time_split(df, test_size)

    expected_train = round(len(df) * (1 - test_size))
    assert abs(len(train) - expected_train) <= 1
    assert len(train) + len(test) == len(df)
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_time_split_no_row_dropped_or_duplicated() -> None:
    df = _featured_frame()
    train, test = time_split(df, 0.25)
    assert len(train) + len(test) == len(df)
    combined_ids = set(train["transaction_id"]) | set(test["transaction_id"])
    assert len(combined_ids) == len(df)


def test_run_generate_data_writes_csv(tmp_path: Path) -> None:
    config = Config(n_samples=200, random_seed=5)
    output = tmp_path / "nested" / "transactions.csv"
    path = run_generate_data(config, output_path=str(output))

    assert path == str(output)
    assert output.exists()
    written = pd.read_csv(output)
    assert len(written) == 200


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_run_train_end_to_end(tmp_path: Path, model_type: str) -> None:
    config = replace(
        Config(model_type=model_type),
        n_samples=2000,
        random_seed=13,
        model_card_path=str(tmp_path / "model_card.md"),
    )
    data_path = tmp_path / "transactions.csv"
    model_path = tmp_path / "model.joblib"

    run_generate_data(config, output_path=str(data_path))
    metrics = run_train(config, data_path=str(data_path), model_path=str(model_path))

    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert loaded is not None

    assert Path(config.model_card_path).exists()

    # PR-AUC must beat a random/constant classifier and not be trivially
    # separable (SPEC Section 4's noise requirement, QA_PLAN end-to-end
    # bound).
    assert metrics["pr_auc"] > config.fraud_rate
    assert metrics["pr_auc"] < 0.999


def test_run_evaluate_never_retrains_and_matches_train_metrics(tmp_path: Path) -> None:
    config = replace(
        Config(model_type="logistic_regression"),
        n_samples=2000,
        random_seed=17,
        model_card_path=str(tmp_path / "model_card.md"),
    )
    data_path = tmp_path / "transactions.csv"
    model_path = tmp_path / "model.joblib"

    run_generate_data(config, output_path=str(data_path))
    train_metrics = run_train(config, data_path=str(data_path), model_path=str(model_path))
    eval_metrics = run_evaluate(config, data_path=str(data_path), model_path=str(model_path))

    assert eval_metrics["pr_auc"] == train_metrics["pr_auc"]
    assert eval_metrics["confusion_matrix"] == train_metrics["confusion_matrix"]


def test_run_train_missing_data_file_raises(tmp_path: Path) -> None:
    config = Config()
    with pytest.raises(FileNotFoundError):
        run_train(config, data_path=str(tmp_path / "missing.csv"), model_path=str(tmp_path / "m.joblib"))


def test_run_evaluate_missing_model_file_raises(tmp_path: Path) -> None:
    config = Config(n_samples=200)
    data_path = tmp_path / "transactions.csv"
    run_generate_data(config, output_path=str(data_path))
    with pytest.raises(FileNotFoundError):
        run_evaluate(config, data_path=str(data_path), model_path=str(tmp_path / "missing.joblib"))
