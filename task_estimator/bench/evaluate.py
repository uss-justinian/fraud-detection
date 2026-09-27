"""Compare estimates with real agent runs, then calibrate.

    python bench/evaluate.py [--actuals bench/results/actuals.jsonl]

Steps:
  1. Rate every sample task with the LLM rubric (cached in results/rubric.json).
  2. Score the uncalibrated priors against the actual runs.
  3. Leave-one-task-out: fit the flow parameters on the other tasks, predict the held-out one.
  4. Compare with baselines and the run-to-run noise floor.
  5. Fit on all runs, save estimator/weights/flow_params.json, write results/report.md.
"""

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from estimator.model import Params, cost, difficulty, estimate, log_turns_mean, tokens_for_turns  # noqa: E402
from estimator.rubric import rate  # noqa: E402

RES = ROOT / "bench" / "results"
QTY = {"turns": "num_turns", "input_tokens": "input_total", "cost_usd": "cost_usd", "minutes": "minutes"}


def load_rubrics(tasks):
    path = RES / "rubric.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for t in tasks:
        if t["id"] not in cache:
            cache[t["id"]] = rate(t["task"], ROOT / "bench" / "sandbox")
            path.write_text(json.dumps(cache, indent=1))
    return cache


def load_actuals(path):
    runs = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("num_turns"):
            r["minutes"] = r["duration_s"] / 60
            runs[r["id"]].append(r)
    return runs


def fit(ids, rub, runs, prior=None, lam=1.0):
    """Fit flow parameters on the given task ids. Ridge-shrinks the turn model towards the priors."""
    prior = prior or Params()
    X, y = [], []
    for i in ids:
        _, z = difficulty(rub[i])
        for r in runs[i]:
            X.append([1, math.log2(max(rub[i]["human_minutes"], 1)), z])
            y.append(math.log(r["num_turns"]))
    X, y = np.array(X), np.array(y)
    b0 = np.array([prior.a0, prior.a1, prior.a2])
    beta = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y + lam * b0)
    resid = y - X @ beta
    all_runs = [r for i in ids for r in runs[i]]
    T = np.array([r["num_turns"] for r in all_runs], float)
    (C0, g), _ = nnls(np.column_stack([T, T * T / 2]), np.array([r["input_total"] for r in all_runs], float))
    p = replace(prior, a0=beta[0], a1=beta[1], a2=beta[2],
                sigma=max(float(np.sqrt(np.mean(resid ** 2) * len(y) / max(len(y) - 3, 1))), 0.25),
                C0=C0, g=g,
                o=sum(r["output"] for r in all_runs) / T.sum(),
                s=float(np.median([r["duration_s"] / r["num_turns"] for r in all_runs])),
                source=f"calibrated on {len(ids)} tasks / {len(all_runs)} runs")
    # cache-write share of the starting context: pick the value that best explains billed cost
    best = min(np.linspace(0, 1, 21), key=lambda f: sum(
        math.log(cost(tokens_for_turns(r["num_turns"], replace(p, c0_write_frac=f)), p) / r["cost_usd"]) ** 2
        for r in all_runs))
    return replace(p, c0_write_frac=float(best))


def score(preds, runs):
    """preds: id -> estimate dict. Returns metrics per quantity against per-task medians."""
    out = {}
    for q, k in QTY.items():
        ids = sorted(preds)
        act = [st.median(r[k] for r in runs[i]) for i in ids]
        p50 = [preds[i]["p50"][q] for i in ids]
        err = [abs(math.log10(p / a)) for p, a in zip(p50, act)]
        cover = [preds[i]["p10"][q] <= r[k] <= preds[i]["p90"][q] for i in ids for r in runs[i]]
        out[q] = {"median_factor_off": round(10 ** st.median(err), 2),
                  "within_2x": f"{sum(e <= math.log10(2) for e in err)}/{len(err)}",
                  "spearman": round(spearmanr(p50, act)[0], 2),
                  "p10_p90_coverage": f"{sum(cover)}/{len(cover)}",
                  "bias_pred_over_actual": round(10 ** st.median(math.log10(p / a) for p, a in zip(p50, act)), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuals", default=str(RES / "actuals.jsonl"))
    a = ap.parse_args()
    tasks = json.loads((ROOT / "bench" / "tasks.json").read_text())
    runs = load_actuals(a.actuals)
    tasks = [t for t in tasks if t["id"] in runs]
    ids = [t["id"] for t in tasks]
    rub = load_rubrics(tasks)

    prior_preds = {i: estimate(rub[i], Params()) for i in ids}
    loo_preds = {i: estimate(rub[i], fit([j for j in ids if j != i], rub, runs)) for i in ids}

    # Baselines, also leave-one-out
    def med_turns(excl):
        return st.median(r["num_turns"] for j in ids if j != excl for r in runs[j])
    turns_act = [st.median(r["num_turns"] for r in runs[i]) for i in ids]
    base = {
        "global median (LOO)": [med_turns(i) for i in ids],
        "LLM guesses its own turns": [rub[i]["agent_turns_guess"] for i in ids],
        "priors (no calibration)": [prior_preds[i]["p50"]["turns"] for i in ids],
        "calibrated (LOO)": [loo_preds[i]["p50"]["turns"] for i in ids],
    }
    base_tbl = {k: {"median_factor_off": round(10 ** st.median(abs(math.log10(p / a)) for p, a in zip(v, turns_act)), 2),
                    "spearman": round(spearmanr(v, turns_act)[0], 2)}
                for k, v in base.items()}
    # A leave-one-out median ranks tasks backwards by construction, so its rank correlation means nothing.
    base_tbl["global median (LOO)"]["spearman"] = None

    # Noise floor: how far apart are two runs of the same task?
    pairs = [(runs[i][0]["num_turns"], runs[i][1]["num_turns"], runs[i][0]["input_total"], runs[i][1]["input_total"])
             for i in ids if len(runs[i]) >= 2]
    noise = {"tasks_with_repeats": len(pairs)}
    if pairs:
        noise["turns_median_factor_between_runs"] = round(10 ** st.median(abs(math.log10(x / y)) for x, y, _, _ in pairs), 2)
        noise["tokens_median_factor_between_runs"] = round(10 ** st.median(abs(math.log10(x / y)) for _, _, x, y in pairs), 2)
        noise["tokens_max_factor_between_runs"] = round(max(max(x, y) / min(x, y) for _, _, x, y in pairs), 2)

    # Does each signal rank tasks the way reality does?
    signals = {"difficulty (psychometrics)": [difficulty(rub[i])[0] for i in ids],
               "human_minutes (LLM)": [rub[i]["human_minutes"] for i in ids],
               "files_touched (LLM)": [rub[i]["files_touched"] for i in ids],
               "agent_turns_guess (LLM)": [rub[i]["agent_turns_guess"] for i in ids]}
    relevance = {k: round(spearmanr(v, turns_act)[0], 2) for k, v in signals.items()}

    # Is the snowball formula the right shape? Fit on actual turns, check tokens.
    full = fit(ids, rub, runs)
    all_runs = [r for i in ids for r in runs[i]]
    tok_pred = [tokens_for_turns(r["num_turns"], full)["input_total"] for r in all_runs]
    snowball = {"median_factor_off_given_true_turns": round(10 ** st.median(
        abs(math.log10(p / r["input_total"])) for p, r in zip(tok_pred, all_runs)), 2),
        "C0": int(full.C0), "g_per_turn": int(full.g), "output_per_turn": int(full.o), "sec_per_turn": round(full.s, 1)}
    success = {"agent_reported_success": f"{sum(r['agent_ok'] for r in all_runs)}/{len(all_runs)}",
               "tests_pass_after": f"{sum(r['tests_pass'] for r in all_runs)}/{len(all_runs)}",
               "note": "the sandbox ships with a failing test, so tests only pass after tasks that fix it"}

    full.fit_stats = {"loo": score(loo_preds, runs), "prior": score(prior_preds, runs)}
    full.save()

    report = {"prior": score(prior_preds, runs), "calibrated_loo": score(loo_preds, runs),
              "turn_baselines": base_tbl, "noise_floor": noise, "signal_vs_actual_turns_spearman": relevance,
              "snowball_formula_check": snowball, "success": success, "fitted_params": {
                  k: round(v, 3) if isinstance(v, float) else v for k, v in vars(full).items() if k != "fit_stats"}}
    per_task = []
    for i in ids:
        rs = runs[i]
        per_task.append({"id": i, "human_min": rub[i]["human_minutes"], "difficulty": prior_preds[i]["difficulty"],
                         "turns_actual": [r["num_turns"] for r in rs],
                         "turns_pred_p10_p50_p90": [loo_preds[i][q]["turns"] for q in ("p10", "p50", "p90")],
                         "tokens_actual_k": [round(r["input_total"] / 1000) for r in rs],
                         "tokens_pred_p50_k": round(loo_preds[i]["p50"]["input_tokens"] / 1000),
                         "cost_actual": [round(r["cost_usd"], 3) for r in rs], "cost_pred_p50": loo_preds[i]["p50"]["cost_usd"],
                         "min_actual": [round(r["minutes"], 1) for r in rs], "min_pred_p50": loo_preds[i]["p50"]["minutes"]})
    report["per_task"] = per_task
    (RES / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
