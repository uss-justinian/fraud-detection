"""Fit the ticket-text -> difficulty model on Agent Psychometrics data.

Uses only the rubric features an LLM can judge from the task text alone, so the
model applies to a new ticket. Writes estimator/weights/difficulty.json.

    python scripts/train_difficulty.py /path/to/agent-psychometrics
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["solution_hint", "domain_knowledge_required", "logical_reasoning_required", "atypicality",
            "verification_difficulty", "error_specificity", "debugging_complexity", "codebase_scope",
            "similar_issue_likelihood", "side_effect_risk"]
BENCHMARKS = {"swebench_verified": "items_verified.csv", "swebench_pro": "items_pro.csv"}


def main(repo):
    repo = Path(repo)
    frames = []
    for bench, items in BENCHMARKS.items():
        b = pd.read_csv(repo / "data/all_benchmarks/1d_1pl" / items, index_col=0)
        f = pd.read_csv(repo / "llm_judge_features/defaults" / bench / "llm_judge_features.csv", index_col=0)
        frames.append(f[FEATURES].join(b["b"], how="inner"))
    d = pd.concat(frames)
    X, y = d[FEATURES], d["b"]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    pred = cross_val_predict(model, X, y, cv=5)
    r = np.corrcoef(pred, y)[0, 1]
    model.fit(X, y)
    sc, rg = model[0], model[1]
    out = {
        "features": FEATURES,
        "mean": dict(zip(FEATURES, sc.mean_.tolist())),
        "scale": dict(zip(FEATURES, sc.scale_.tolist())),
        "coef": dict(zip(FEATURES, rg.coef_.tolist())),
        "intercept": float(rg.intercept_),
        "difficulty_mean": float(y.mean()), "difficulty_std": float(y.std()),
        "cv_pearson_r": float(r), "n_tasks": int(len(d)),
        "source": "Agent Psychometrics (arXiv 2604.00594), 1PL IRT difficulties, MIT licence",
    }
    dest = Path(__file__).resolve().parent.parent / "estimator/weights/difficulty.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"n={len(d)} cv r={r:.2f} -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1])
