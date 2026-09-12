# fraud-detection

An enterprise-scope fraud-detection platform **architecture**, with one
component built end-to-end as a working reference implementation — the
first project in a series exploring how AI agents build real systems when
given a real architecture to build against, not just a one-line prompt.

## Start here

1. **[`docs/architecture/`](docs/architecture/)** — the platform
   architecture: system context, components, data flows, detection
   strategy, decisioning, governance, compliance considerations, and a
   glossary of domain terms. This is the design, and it's bigger than
   what's currently running as code — [`DOMAIN_ARCHITECTURE.md`'s
   Build Status Map](docs/architecture/DOMAIN_ARCHITECTURE.md#14-build-status-map)
   says exactly which is which.
2. **[`docs/architecture/CONTEXT_ARCHITECTURE.md`](docs/architecture/CONTEXT_ARCHITECTURE.md)**
   — how AI agents are given exactly the context needed to build one
   component of that architecture correctly and consistently, without
   drift between components or between projects. This is the reusable
   "how AI enables development" part of the series, independent of fraud
   detection specifically.
3. **[`docs/process/`](docs/process/)** — the concrete execution mechanism
   (Specifier → Implementor → Reviewer) that item 2 above describes in
   the abstract; used to build the one component that exists today.

Every claim is backed by a committed artifact, not a narrative written
after the fact:

| Layer | Artifact | What it proves |
|---|---|---|
| Domain architecture | [`docs/architecture/DOMAIN_ARCHITECTURE.md`](docs/architecture/DOMAIN_ARCHITECTURE.md) | The platform this component is one piece of, and why it's shaped the way it is |
| Component spec (Specifier) | [`docs/spec/SPEC.md`](docs/spec/SPEC.md), [`docs/spec/QA_PLAN.md`](docs/spec/QA_PLAN.md) | Scope, design decisions, and acceptance criteria for this one component, written *before* any code existed |
| Implementation (Implementor) | [`src/fraud_detection/`](src/fraud_detection/), [`tests/`](tests/) | Code built strictly against the spec above |
| Review (Reviewer/QA) | [`docs/qa/QA_REPORT.md`](docs/qa/QA_REPORT.md) | Independent re-run of every quality gate and acceptance criterion, in a fresh environment, without trusting the Implementor's self-report |

## The one component built so far: Offline Model Training & Batch Evaluation

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
