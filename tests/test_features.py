"""Tests for fraud_detection.features (SPEC Section 5 / QA_PLAN
feature-engineering acceptance criteria)."""

import numpy as np
import pandas as pd

from fraud_detection.config import Config
from fraud_detection.features import add_features


def _base_row(**overrides: object) -> dict:
    row = {
        "transaction_id": 0,
        "customer_id": 1,
        "timestamp": pd.Timestamp("2024-01-01 12:00:00"),
        "amount": 100.0,
        "account_age_days": 365,
        "distance_from_home_km": 5.0,
        "merchant_category": "grocery",
        "is_foreign": False,
        "is_fraud": False,
    }
    row.update(overrides)
    return row


def test_does_not_mutate_input() -> None:
    df = pd.DataFrame([_base_row(), _base_row(transaction_id=1)])
    before = df.copy(deep=True)
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_transactions_last_24h_exact_fixture() -> None:
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    df = pd.DataFrame(
        [
            _base_row(transaction_id=0, timestamp=t0),
            _base_row(transaction_id=1, timestamp=t0 + pd.Timedelta(hours=1)),
            _base_row(transaction_id=2, timestamp=t0 + pd.Timedelta(hours=25)),
        ]
    )
    result = add_features(df)
    assert result["transactions_last_24h"].tolist() == [0, 1, 0]


def test_transactions_last_24h_excludes_other_customers() -> None:
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    df = pd.DataFrame(
        [
            _base_row(transaction_id=0, customer_id=1, timestamp=t0),
            _base_row(
                transaction_id=1,
                customer_id=2,
                timestamp=t0 + pd.Timedelta(minutes=30),
            ),
            _base_row(
                transaction_id=2,
                customer_id=1,
                timestamp=t0 + pd.Timedelta(hours=1),
            ),
        ]
    )
    result = add_features(df)
    # Row 2 (customer 1) should only count row 0 (customer 1), not row 1
    # (customer 2).
    assert result["transactions_last_24h"].tolist() == [0, 0, 1]


def test_amount_log_hour_weekend_night_fixture() -> None:
    # 2024-01-06 is a Saturday (weekend); 03:00 is night; 2024-01-08 is a
    # Monday (weekday); 14:00 is day.
    weekend_night = pd.Timestamp("2024-01-06 03:00:00")
    weekday_day = pd.Timestamp("2024-01-08 14:00:00")
    df = pd.DataFrame(
        [
            _base_row(transaction_id=0, amount=100.0, timestamp=weekend_night),
            _base_row(transaction_id=1, amount=0.0, timestamp=weekday_day),
        ]
    )
    result = add_features(df)

    assert np.isclose(result.loc[0, "amount_log"], np.log1p(100.0))
    assert np.isclose(result.loc[1, "amount_log"], np.log1p(0.0))

    assert result.loc[0, "hour_of_day"] == 3
    assert result.loc[1, "hour_of_day"] == 14

    assert bool(result.loc[0, "is_weekend"]) is True
    assert bool(result.loc[1, "is_weekend"]) is False

    assert bool(result.loc[0, "is_night"]) is True
    assert bool(result.loc[1, "is_night"]) is False


def test_is_night_boundary_inclusive_of_hour_5() -> None:
    df = pd.DataFrame(
        [
            _base_row(transaction_id=0, timestamp=pd.Timestamp("2024-01-01 05:00:00")),
            _base_row(transaction_id=1, timestamp=pd.Timestamp("2024-01-01 06:00:00")),
        ]
    )
    result = add_features(df)
    assert bool(result.loc[0, "is_night"]) is True
    assert bool(result.loc[1, "is_night"]) is False


def test_output_contains_all_configured_feature_columns() -> None:
    df = pd.DataFrame([_base_row(), _base_row(transaction_id=1)])
    result = add_features(df)
    config = Config()
    for column in config.numeric_features + config.categorical_features:
        assert column in result.columns
