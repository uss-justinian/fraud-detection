"""Refit estimate.py's constants from real runs and report how far off it is.

    python bench/run_flows.py --reps 2       # real agent runs -> bench/results/actuals.jsonl
    python calibrate.py                      # ratings -> bench/results/ratings.json, fit -> params.json
"""

import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from estimate import DEFAULTS, PARAMS_FILE, predict, rate

HERE = Path(__file__).resolve().parent
RES = HERE / "bench" / "results"


def features(r):
    return [1, math.log2(max(r["minutes"], 1)), math.log2(r["files"] + 1)]


def fit(runs, ratings):
    A = np.array([features(ratings[r["id"]]) for r in runs])
    y = np.log([r["num_turns"] for r in runs])
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    T = np.array([r["num_turns"] for r in runs], float)
    C0, g = np.linalg.lstsq(np.column_stack([T, T * T / 2]), [r["input_total"] for r in runs], rcond=None)[0]
    k_in, k_out = np.linalg.lstsq(np.array([[r["input_total"], r["output"]] for r in runs]),
                                  [r["cost_usd"] for r in runs], rcond=None)[0] * 1e6
    resid = y - A @ [a, b, c]
    return {**DEFAULTS, "a": a, "b": b, "c": c, "C0": C0, "g": g,
            "sigma": max(float(resid.std(ddof=3)), 0.35),  # floor: a small benchmark understates spread
            "out_per_turn": sum(r["output"] for r in runs) / T.sum(),
            "sec_per_turn": st.median(r["duration_s"] / r["num_turns"] for r in runs),
            "k_in": k_in, "k_out": k_out}


def main():
    tasks = {t["id"]: t["task"] for t in json.loads((HERE / "bench" / "tasks.json").read_text())}
    runs = [json.loads(line) for line in open(RES / "actuals.jsonl")]
    runs = [r for r in runs if r.get("num_turns")]
    path = RES / "ratings.json"
    ratings = json.loads(path.read_text()) if path.exists() else {}
    for i in {r["id"] for r in runs} - set(ratings):
        ratings[i] = rate(tasks[i], HERE / "bench" / "sandbox")
        path.write_text(json.dumps(ratings, indent=1))

    # Leave one task out: predict each task from a fit on the others.
    rows = []
    for i in sorted(ratings):
        p = fit([r for r in runs if r["id"] != i], ratings)
        est = predict(ratings[i]["minutes"], ratings[i]["files"], p)
        mine = [r for r in runs if r["id"] == i]
        act = {"turns": st.median(r["num_turns"] for r in mine),
               "tokens": st.median(r["input_total"] + r["output"] for r in mine),
               "cost_usd": st.median(r["cost_usd"] for r in mine)}
        rows.append((i, est, act))
        print(f"{i:16} turns {est['p50']['turns']:5} vs {act['turns']:4}   "
              f"cost ${est['p50']['cost_usd']:.3f} vs ${act['cost_usd']:.3f}")
    for q in ("turns", "tokens", "cost_usd"):
        err = [abs(math.log10(e["p50"][q] / a[q])) for _, e, a in rows]
        print(f"{q:8}: typically {10 ** st.median(err):.2f}x off, "
              f"{sum(x <= math.log10(2) for x in err)}/{len(err)} within 2x")

    p = fit(runs, ratings)
    PARAMS_FILE.write_text(json.dumps({k: round(float(v), 4) for k, v in p.items()}, indent=2))
    print(f"saved {PARAMS_FILE.name}")


if __name__ == "__main__":
    main()
