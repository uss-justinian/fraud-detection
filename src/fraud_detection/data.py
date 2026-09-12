"""Synthetic transaction data generation (SPEC Section 4).

Pure, deterministic, no file I/O — all randomness is drawn from a
``numpy.random.default_rng(config.random_seed)`` instance so that
``generate_transactions`` is fully reproducible for a fixed seed. File I/O
(reading/writing CSVs) lives in ``pipeline.py``, per SPEC Section 12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import Config

MERCHANT_CATEGORIES: list[str] = [
    "grocery",
    "electronics",
    "travel",
    "restaurant",
    "online",
    "gas_station",
    "entertainment",
    "other",
]

# Fixed anchor for the end of the generated data window. A constant (rather
# than the wall-clock "now") keeps generate_transactions fully deterministic
# for a given random_seed, per SPEC Section 4 ("output is fully reproducible
# for a fixed seed").
_WINDOW_END = pd.Timestamp("2024-01-01 00:00:00")

# Internal, fixed proportions of the three fraud sub-populations (SPEC
# Section 4). These are intentionally not exposed via Config.
_FRAUD_FOREIGN_HIGH_VALUE_FRAC = 0.40
_FRAUD_VELOCITY_BURST_FRAC = 0.40
# Remaining ~0.20 of fraud rows are "noise" (see _generate_fraud).

# Fraction of *legitimate* rows generated with foreign+high-value
# characteristics (real business travel), overlapping fraud pattern 1.
_LEGIT_BUSINESS_TRAVEL_FRAC = 0.02

_RAW_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "account_age_days",
    "distance_from_home_km",
    "merchant_category",
    "is_foreign",
    "is_fraud",
]


def generate_transactions(config: Config) -> pd.DataFrame:
    """Generate a raw, timestamp-sorted synthetic transaction DataFrame.

    Row count is ``config.n_samples``; fraud rows are
    ``round(n_samples * config.fraud_rate)`` of the total, built from three
    mixed sub-populations plus a small overlapping slice of legitimate rows,
    so that fraud is not trivially separable (SPEC Section 4).
    """
    rng = np.random.default_rng(config.random_seed)

    n_samples = config.n_samples
    n_fraud = round(n_samples * config.fraud_rate)
    n_legit = n_samples - n_fraud
    n_customers = max(200, n_samples // 20)

    window_start = _WINDOW_END - pd.Timedelta(days=config.n_days)

    frames = [_generate_legit(rng, n_legit, n_customers, window_start)]
    if n_fraud > 0:
        frames.append(_generate_fraud(rng, n_fraud, n_customers, window_start))

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("timestamp", ascending=True, kind="mergesort")
    full = full.reset_index(drop=True)
    full["transaction_id"] = np.arange(len(full))

    return full[_RAW_COLUMNS]


def _uniform_timestamps(
    rng: np.random.Generator, n: int, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> pd.Series:
    """n timestamps drawn uniformly over [window_start, window_end)."""
    span_seconds = (window_end - window_start).total_seconds()
    offsets = rng.uniform(0.0, span_seconds, size=n)
    return pd.Series(window_start + pd.to_timedelta(offsets, unit="s"))


def _generate_base(
    rng: np.random.Generator,
    n: int,
    n_customers: int,
    window_start: pd.Timestamp,
    is_fraud_label: bool,
) -> pd.DataFrame:
    """The "ordinary" transaction distribution shared by legitimate rows and
    the fraud "noise" sub-population (SPEC Section 4, pattern 3), which is
    sampled from these same distributions and simply labeled fraud.
    """
    customer_ids = rng.integers(0, n_customers, size=n)
    timestamps = _uniform_timestamps(rng, n, window_start, _WINDOW_END)
    amount = rng.lognormal(mean=3.5, sigma=0.9, size=n)
    account_age_days = rng.integers(1, 3650, size=n)
    distance_from_home_km = rng.exponential(scale=5.0, size=n)
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n)
    is_foreign = rng.random(n) < 0.02

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "timestamp": timestamps,
            "amount": amount,
            "account_age_days": account_age_days,
            "distance_from_home_km": distance_from_home_km,
            "merchant_category": pd.Categorical(merchant_category, categories=MERCHANT_CATEGORIES),
            "is_foreign": is_foreign,
            "is_fraud": np.full(n, is_fraud_label, dtype=bool),
        }
    )


def _generate_legit(
    rng: np.random.Generator, n: int, n_customers: int, window_start: pd.Timestamp
) -> pd.DataFrame:
    df = _generate_base(rng, n, n_customers, window_start, False)

    # ~2% "real business travel": foreign + high value, overlapping the
    # feature distribution of fraud pattern 1 so that "any foreign
    # high-value transaction" is not itself a perfect fraud signature.
    n_business_travel = round(n * _LEGIT_BUSINESS_TRAVEL_FRAC)
    if n_business_travel > 0:
        travel_idx = rng.choice(n, size=n_business_travel, replace=False)
        df.loc[travel_idx, "is_foreign"] = True
        df.loc[travel_idx, "amount"] = rng.lognormal(mean=5.3, sigma=1.0, size=n_business_travel)
        df.loc[travel_idx, "distance_from_home_km"] = rng.uniform(
            500.0, 8000.0, size=n_business_travel
        )
    return df


def _generate_fraud(
    rng: np.random.Generator, n: int, n_customers: int, window_start: pd.Timestamp
) -> pd.DataFrame:
    n_type1 = round(n * _FRAUD_FOREIGN_HIGH_VALUE_FRAC)
    n_type2 = round(n * _FRAUD_VELOCITY_BURST_FRAC)
    n_type3 = n - n_type1 - n_type2
    if n_type3 < 0:  # pragma: no cover - unreachable while both fractions are 0.4
        # Defensive guard in case the fixed proportions above are ever
        # changed to sum to more than 1.0: borrow back from type2 rather
        # than passing a negative count downstream.
        n_type2 += n_type3
        n_type3 = 0

    parts = []
    if n_type1 > 0:
        parts.append(_generate_fraud_foreign_high_value(rng, n_type1, n_customers, window_start))
    if n_type2 > 0:
        parts.append(_generate_fraud_velocity_burst(rng, n_type2, n_customers, window_start))
    if n_type3 > 0:
        parts.append(_generate_base(rng, n_type3, n_customers, window_start, True))
    return pd.concat(parts, ignore_index=True)


def _generate_fraud_foreign_high_value(
    rng: np.random.Generator, n: int, n_customers: int, window_start: pd.Timestamp
) -> pd.DataFrame:
    """Fraud pattern 1: foreign + high-value + new-account (SPEC Section 4)."""
    customer_ids = rng.integers(0, n_customers, size=n)
    timestamps = _uniform_timestamps(rng, n, window_start, _WINDOW_END)
    amount = rng.lognormal(mean=5.6, sigma=1.0, size=n)
    account_age_days = rng.integers(0, 31, size=n)  # uniform in [0, 30]
    distance_from_home_km = rng.uniform(500.0, 10000.0, size=n)

    # Weighted toward electronics/travel/online.
    weights = np.array([0.05, 0.25, 0.25, 0.05, 0.25, 0.05, 0.05, 0.05])
    weights = weights / weights.sum()
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n, p=weights)

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "timestamp": timestamps,
            "amount": amount,
            "account_age_days": account_age_days,
            "distance_from_home_km": distance_from_home_km,
            "merchant_category": pd.Categorical(merchant_category, categories=MERCHANT_CATEGORIES),
            "is_foreign": np.ones(n, dtype=bool),
            "is_fraud": np.ones(n, dtype=bool),
        }
    )


def _generate_fraud_velocity_burst(
    rng: np.random.Generator, n: int, n_customers: int, window_start: pd.Timestamp
) -> pd.DataFrame:
    """Fraud pattern 2: card-testing / velocity bursts (SPEC Section 4).

    Generated as bursts of several transactions for the same customer_id
    within a short (minutes) window, at unusual hours (forced into 0-5 at
    the raw-timestamp level), moderate amounts, and distance_from_home_km
    well above the legitimate distribution.
    """
    span_days = max((_WINDOW_END - window_start).days, 1)
    rows = []
    remaining = n
    while remaining > 0:
        burst_size = min(remaining, int(rng.integers(3, 7)))
        customer_id = int(rng.integers(0, n_customers))

        day_offset = int(rng.integers(0, span_days))
        base_day = window_start + pd.Timedelta(days=day_offset)
        hour = int(rng.integers(0, 6))  # unusual hour, forced into [0, 5]
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        base_time = pd.Timestamp(
            year=base_day.year,
            month=base_day.month,
            day=base_day.day,
            hour=hour,
            minute=minute,
            second=second,
        )
        offsets_minutes = np.sort(rng.uniform(0.0, 15.0, size=burst_size))
        timestamps = [base_time + pd.Timedelta(minutes=float(m)) for m in offsets_minutes]

        amount = rng.lognormal(mean=4.0, sigma=0.7, size=burst_size)
        account_age_days = rng.integers(1, 3650, size=burst_size)
        distance_from_home_km = rng.uniform(200.0, 3000.0, size=burst_size)
        merchant_category = rng.choice(MERCHANT_CATEGORIES, size=burst_size)
        is_foreign = rng.random(burst_size) < 0.3

        rows.append(
            pd.DataFrame(
                {
                    "customer_id": np.full(burst_size, customer_id),
                    "timestamp": timestamps,
                    "amount": amount,
                    "account_age_days": account_age_days,
                    "distance_from_home_km": distance_from_home_km,
                    "merchant_category": pd.Categorical(
                        merchant_category, categories=MERCHANT_CATEGORIES
                    ),
                    "is_foreign": is_foreign,
                    "is_fraud": np.ones(burst_size, dtype=bool),
                }
            )
        )
        remaining -= burst_size

    return pd.concat(rows, ignore_index=True)
