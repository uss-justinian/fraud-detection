"""Tests for fraud_detection.data (SPEC Section 4 / QA_PLAN data-generation
acceptance criteria)."""

from dataclasses import replace

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data import generate_transactions

_RAW_COLUMNS = {
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "account_age_days",
    "distance_from_home_km",
    "merchant_category",
    "is_foreign",
    "is_fraud",
}


def _small_config(**overrides: object) -> Config:
    base = Config(n_samples=500, random_seed=7)
    return replace(base, **overrides)


def test_returns_expected_row_count_and_schema() -> None:
    config = _small_config()
    df = generate_transactions(config)
    assert len(df) == config.n_samples
    assert set(df.columns) == _RAW_COLUMNS


def test_reproducible_for_fixed_seed() -> None:
    config = _small_config()
    df1 = generate_transactions(config)
    df2 = generate_transactions(config)
    pd.testing.assert_frame_equal(df1, df2)


def test_timestamp_is_non_decreasing() -> None:
    config = _small_config()
    df = generate_transactions(config)
    assert df["timestamp"].is_monotonic_increasing


def test_realized_fraud_rate_within_tolerance() -> None:
    config = _small_config(n_samples=5000, fraud_rate=0.02)
    df = generate_transactions(config)
    realized = df["is_fraud"].mean()
    assert abs(realized - config.fraud_rate) <= 0.30 * config.fraud_rate


def test_is_fraud_is_boolean_with_no_nulls() -> None:
    config = _small_config()
    df = generate_transactions(config)
    assert df["is_fraud"].dtype == bool
    assert df["is_fraud"].isna().sum() == 0


def test_customer_pool_size_matches_spec_formula() -> None:
    config = _small_config(n_samples=10_000)
    df = generate_transactions(config)
    expected_pool = max(200, config.n_samples // 20)
    assert df["customer_id"].nunique() <= expected_pool
    assert df["customer_id"].min() >= 0


def test_different_seeds_produce_different_data() -> None:
    df1 = generate_transactions(_small_config(random_seed=1))
    df2 = generate_transactions(_small_config(random_seed=2))
    assert not df1["amount"].equals(df2["amount"])


def test_zero_fraud_rate_does_not_raise() -> None:
    config = _small_config(fraud_rate=0.0)
    df = generate_transactions(config)
    assert len(df) == config.n_samples
    assert not df["is_fraud"].any()
