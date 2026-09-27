"""Compare estimates with real runs, refit params.json and print accuracy.

    python bench/run_flows.py --reps 2     # real agent runs -> bench/results/actuals.jsonl
    python -m estimator.calibrate
"""

import json
import statistics as st
from dataclasses import asdict
from pathlib import Path

from estimator.fitting import ACTUAL, factor_off, fit, leave_one_out
from estimator.rating import Rating, rate

BENCH = Path(__file__).resolve().parent.parent / "bench"
RESULTS = BENCH / "results"


def load_runs():
    runs = (json.loads(line) for line in open(RESULTS / "actuals.jsonl"))
    return [r for r in runs if r.get("num_turns")]


def load_ratings(task_ids):
    """Cached ratings; rates any task that has none yet."""
    path = RESULTS / "ratings.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    tasks = {t["id"]: t["task"] for t in json.loads((BENCH / "tasks.json").read_text())}
    for task_id in sorted(set(task_ids) - set(cached)):
        cached[task_id] = asdict(rate(tasks[task_id], BENCH / "sandbox"))
        path.write_text(json.dumps(cached, indent=1))
    return {k: Rating(**v) for k, v in cached.items()}


def main():
    runs = load_runs()
    ratings = load_ratings({r["id"] for r in runs})
    rows = leave_one_out(runs, ratings)
    for task_id, est, act in rows:
        print(f"{task_id:16} turns {est['p50']['turns']:5} vs {act['turns']:4}   "
              f"cost ${est['p50']['cost_usd']:.3f} vs ${act['cost_usd']:.3f}")
    for q in ACTUAL:
        off = [factor_off(est["p50"][q], act[q]) for _, est, act in rows]
        print(f"{q:8}: typically {st.median(off):.2f}x off, {sum(x <= 2 for x in off)}/{len(off)} within 2x")
    fit(runs, ratings).save()
    print("saved params.json")


if __name__ == "__main__":
    main()
