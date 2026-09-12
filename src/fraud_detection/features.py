"""Feature engineering (SPEC Section 5).

Pure, no file I/O: ``add_features`` takes the raw, timestamp-sorted schema
produced by ``data.generate_transactions`` and returns a new DataFrame with
derived columns, without mutating its input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_VELOCITY_WINDOW = np.timedelta64(24, "h")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with engineered columns added.

    Adds ``amount_log``, ``hour_of_day``, ``is_weekend``, ``is_night``, and
    ``transactions_last_24h``. Together with the raw schema's
    ``merchant_category``/``is_foreign``, these satisfy every field named in
    ``Config.numeric_features``/``Config.categorical_features``.
    """
    result = df.copy(deep=True)
    result["amount_log"] = np.log1p(result["amount"])
    result["hour_of_day"] = result["timestamp"].dt.hour
    result["is_weekend"] = result["timestamp"].dt.dayofweek >= 5
    result["is_night"] = result["hour_of_day"].between(0, 5)
    result["transactions_last_24h"] = _transactions_last_24h(result)
    return result


def _transactions_last_24h(df: pd.DataFrame) -> pd.Series:
    """For each row, count *other* transactions by the same customer_id with
    timestamp strictly in ``(row.timestamp - 24h, row.timestamp)``.

    Leakage rule (SPEC Section 5): only ever counts transactions strictly
    earlier than the row being scored — never the row itself, and never
    anything at or after its own timestamp.
    """
    counts = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

    for _, group in df.groupby("customer_id", sort=False):
        # Sort this customer's rows chronologically so the two-pointer /
        # searchsorted logic below can assume ascending order, regardless of
        # the row order in the input DataFrame.
        order = np.argsort(group["timestamp"].to_numpy(), kind="mergesort")
        sorted_index = group.index.to_numpy()[order]
        times = group["timestamp"].to_numpy()[order]

        lower_bounds = times - _VELOCITY_WINDOW
        # First index whose timestamp is > lower_bounds[i] (strictly after
        # the start of the 24h window).
        lo = np.searchsorted(times, lower_bounds, side="right")
        # First index whose timestamp equals times[i] (excludes ties with
        # the row itself, satisfying "strictly earlier").
        hi = np.searchsorted(times, times, side="left")

        counts.loc[sorted_index] = hi - lo

    return counts
