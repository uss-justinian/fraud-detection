"""Fit Params from logged runs, and score predictions against them."""

import math
import statistics as st

import numpy as np

from estimator.model import Params, predict, token_basis, turn_features

MIN_SIGMA = 0.35  # a small benchmark understates real spread


def _lstsq(rows, targets):
    return np.linalg.lstsq(np.array(rows, float), np.array(targets, float), rcond=None)[0]


def fit(runs, ratings):
    """runs: dicts from bench/run_flows.py. ratings: task id -> Rating."""
    turns = [r["num_turns"] for r in runs]
    X = [turn_features(ratings[r["id"]]) for r in runs]
    y = np.log(turns)
    a, b, c = _lstsq(X, y)
    resid = y - np.array(X) @ (a, b, c)
    C0, g = _lstsq([token_basis(t) for t in turns], [r["input_total"] for r in runs])
    k_in, k_out = _lstsq([(r["input_total"], r["output"]) for r in runs], [r["cost_usd"] for r in runs]) * 1e6
    return Params(a=a, b=b, c=c, sigma=max(float(resid.std(ddof=3)), MIN_SIGMA), C0=C0, g=g,
                  out_per_turn=sum(r["output"] for r in runs) / sum(turns),
                  sec_per_turn=st.median(r["duration_s"] / r["num_turns"] for r in runs),
                  k_in=k_in, k_out=k_out)


ACTUAL = {"turns": lambda r: r["num_turns"], "tokens": lambda r: r["input_total"] + r["output"],
          "cost_usd": lambda r: r["cost_usd"]}


def leave_one_out(runs, ratings):
    """Predict each task from a fit on the others. Returns (task id, estimate, median actuals)."""
    rows = []
    for task_id in sorted(ratings):
        mine = [r for r in runs if r["id"] == task_id]
        if not mine:
            continue
        est = predict(ratings[task_id], fit([r for r in runs if r["id"] != task_id], ratings))
        rows.append((task_id, est, {q: st.median(map(f, mine)) for q, f in ACTUAL.items()}))
    return rows


def factor_off(predicted, actual):
    return 10 ** abs(math.log10(predicted / actual))
