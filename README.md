# fraud-detection

A synthetic-data fraud-detection ML pipeline — the first project in a
series built and documented in public using a **3-agent AI build process**
(Specifier → Implementor → Reviewer) instead of one agent doing everything
end to end.

Every claim below is backed by a committed artifact in this repo, not a
narrative written after the fact:

| Stage | Artifact | What it proves |
|---|---|---|
| Specifier | [`docs/spec/SPEC.md`](docs/spec/SPEC.md), [`docs/spec/QA_PLAN.md`](docs/spec/QA_PLAN.md) | Scope, design decisions, and acceptance criteria written *before* any code existed |
| Implementor | [`src/fraud_detection/`](src/fraud_detection/), [`tests/`](tests/) | Code built strictly against the spec above |
| Reviewer / QA | [`docs/qa/QA_REPORT.md`](docs/qa/QA_REPORT.md) | Independent re-run of every quality gate and acceptance criterion, in a fresh environment, without trusting the Implementor's self-report |

See [`docs/process/README.md`](docs/process/README.md) for the reusable
role definitions ([`specifier.md`](docs/process/specifier.md),
[`implementor.md`](docs/process/implementor.md),
[`reviewer.md`](docs/process/reviewer.md)) — these carry over unchanged to
the next project in the series; only the spec and QA report are
project-specific.

## What this project is

A local, batch CLI that:

1. Generates synthetic transaction data with three deliberately overlapping
   fraud signatures (not a trivially-separable toy dataset) at a realistic
   ~1.5% base rate.
2. Engineers a leakage-safe feature set, including a backward-only
   transaction-velocity signal.
3. Trains a model (logistic regression or random forest, swappable via
   config, not code) behind a single scikit-learn `Pipeline`, using a
   **time-ordered split** and `class_weight="balanced"` for the class
   imbalance — both chosen for reasons specific to fraud data, documented
   in `SPEC.md`.
4. Evaluates using **PR-AUC and recall-at-fixed-precision**, not
   accuracy/ROC-AUC alone, which are misleading at this imbalance level.
5. Writes a model card documenting data provenance, metrics, and intended
   (non-)use.

**This is a portfolio/educational project.** All data is synthetic —
metrics produced here are not a claim about real-world fraud-detection
performance. That limitation is stated in `SPEC.md`, in the code, and in
every generated model card.

## Quickstart

```bash
pip install -e ".[dev]"

python -m fraud_detection.cli generate-data --n-samples 20000 --seed 42
python -m fraud_detection.cli train --model-type random_forest
python -m fraud_detection.cli evaluate
```

Each command's `--help` documents its flags. Quality gates (same ones CI
and the Reviewer agent run):

```bash
ruff check src tests
mypy src
pytest --cov=fraud_detection --cov-report=term-missing --cov-fail-under=80
```

## Result snapshot

From the Reviewer's independently-run end-to-end check (2000 synthetic
rows, seed 42): logistic regression reaches `pr_auc≈0.50`, random forest
reaches `pr_auc≈0.84` — both comfortably above the ~1.5% no-skill baseline
and below 1.0, confirming the synthetic fraud patterns aren't trivially
separable. Full numbers and how they were derived are in
[`docs/qa/QA_REPORT.md`](docs/qa/QA_REPORT.md).
