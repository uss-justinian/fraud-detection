# QA_PLAN: Fraud Detection ML Pipeline

This plan is checked against `docs/spec/SPEC.md`. Every acceptance
criterion below is binary and must be verified by reading code, running a
command, or reading test output.

## 1. Acceptance Criteria

### Data generation (`data.py`)
- [ ] `generate_transactions(config)` returns a DataFrame with exactly
      `config.n_samples` rows and the raw schema columns listed in SPEC
      Section 4.
- [ ] With a fixed `random_seed`, two calls to `generate_transactions`
      produce byte-identical DataFrames (`pd.testing.assert_frame_equal`).
- [ ] The `timestamp` column is non-decreasing (`df.timestamp.is_monotonic_increasing`).
- [ ] The realized fraud rate (`is_fraud.mean()`) is within ±30% relative
      of `config.fraud_rate` for `n_samples >= 5000`.
- [ ] `is_fraud` is a boolean/0-1 column with no nulls.

### Feature engineering (`features.py`)
- [ ] `add_features` does not mutate its input DataFrame (input is
      unchanged after the call, verified by comparing to a pre-call copy).
- [ ] On a hand-built 3-row DataFrame for one `customer_id` at times `t0`,
      `t0 + 1h`, `t0 + 25h`, `transactions_last_24h` equals `[0, 1, 0]`
      exactly.
- [ ] `amount_log`, `hour_of_day`, `is_weekend`, `is_night` match hand-computed
      values on a small fixture for at least one weekend/weekday and one
      night/day timestamp each.
- [ ] Output contains every column named in `Config.numeric_features` and
      `Config.categorical_features`.

### Split (`pipeline.py`)
- [ ] `time_split(df, test_size)` returns train/test partitions whose sizes
      match `round(len(df) * (1 - test_size))` / remainder (±1 for rounding).
- [ ] `max(train.timestamp) <= min(test.timestamp)`.
- [ ] No row appears in both partitions and no row is dropped
      (`len(train) + len(test) == len(df)`).

### Model (`model.py`)
- [ ] `build_pipeline(config)` returns an unfitted `sklearn.pipeline.Pipeline`.
- [ ] For `model_type in {"logistic_regression", "random_forest"}`, the
      final estimator's `class_weight` attribute equals `"balanced"`.
- [ ] An invalid `model_type` (e.g. `"svm"`) raises `ValueError`.
- [ ] Both model types can `.fit()` and `.predict_proba()` on a small
      synthetic frame built from `generate_transactions` + `add_features`
      without raising.

### Evaluation (`evaluate.py`)
- [ ] On a hand-built `y_true`/`y_prob` pair with a known precision-recall
      curve, `compute_metrics` returns `pr_auc`/`roc_auc` matching
      `sklearn.metrics.average_precision_score`/`roc_auc_score` computed
      independently in the test.
- [ ] `recall_at_precision`/`threshold` match a manually-derived value from
      `precision_recall_curve` on the same fixture.
- [ ] When no threshold reaches `target_precision` (e.g. all `y_prob`
      identical), `compute_metrics` returns `threshold=None`,
      `recall_at_precision=None`, and does not raise.
- [ ] `confusion_matrix` keys are exactly `{tn, fp, fn, tp}` and values sum
      to `len(y_true)`.
- [ ] `write_model_card` produces a file containing (case-insensitive
      substring match) the words "synthetic", the configured `model_type`,
      and the computed `pr_auc` value rendered as text.

### CLI / pipeline integration
- [ ] `python -m fraud_detection.cli generate-data --n-samples 500 --seed 1
      --output <tmp>/t.csv` exits 0 and writes a CSV with 500 rows.
- [ ] `python -m fraud_detection.cli train --data-path <tmp>/t.csv
      --model-path <tmp>/m.joblib --model-type logistic_regression` exits 0,
      produces a `joblib.load`-able model at `<tmp>/m.joblib`, and writes a
      model card to `config.model_card_path` (or an override, if the CLI
      exposes one).
- [ ] Repeat the `train` command with `--model-type random_forest`; both
      runs succeed with no code change required.
- [ ] `python -m fraud_detection.cli evaluate --data-path <tmp>/t.csv
      --model-path <tmp>/m.joblib` exits 0 and prints/writes `pr_auc`,
      `roc_auc`, and the confusion matrix.
- [ ] `python -m fraud_detection.cli train --model-type not-a-model`
      exits non-zero with a message naming the valid choices (no stack
      trace to stdout).
- [ ] `python -m fraud_detection.cli --help` and each subcommand's
      `--help` exit 0.
- [ ] End-to-end on a 2000-row generated dataset: `pr_auc` from `train`'s
      test-partition evaluation is strictly greater than `config.fraud_rate`
      (better than a random/constant classifier) and strictly less than
      0.999 (not trivially separable, confirming SPEC Section 4's noise
      requirement).

### Non-scope / dependency guardrails
- [ ] `pyproject.toml` runtime `[project].dependencies` still contains only
      pandas, numpy, scikit-learn, joblib (no new runtime deps added).
- [ ] `grep -r` for `smote`/`imblearn`/`SMOTE` in `src/` returns no matches.
- [ ] No `predict` subcommand exists in `cli.py` (confirms non-goal
      honored) — or if added, treat as a spec deviation to flag, not a pass.

## 2. Test Plan (files expected under `tests/`)

- `tests/test_data.py` — the data-generation assertions above.
- `tests/test_features.py` — the feature-engineering assertions above,
  including the exact backward-window `transactions_last_24h` fixture.
- `tests/test_model.py` — the model-construction assertions above.
- `tests/test_evaluate.py` — the metrics/model-card assertions above,
  including the "target precision unreachable" branch.
- `tests/test_pipeline.py` — `time_split` assertions, plus one end-to-end
  test per `model_type` (small `n_samples`, e.g. 2000, for speed) asserting
  the model artifact, model card, and the PR-AUC bounds above.
- `tests/test_cli.py` — invokes `cli.main([...])` (or subprocess) for each
  subcommand and the error cases above, using `tmp_path` for all file I/O.
- `tests/test_config.py` — `Config()` instantiates with no arguments; the
  three new fields (`target_precision`, `n_days`, `model_card_path`) are
  present with the defaults specified in SPEC Section 13.

All tests must be deterministic (fixed seeds throughout) — no test may rely
on a specific random draw succeeding by chance beyond the documented
tolerance bands above.

## 3. Quality Gates (exact commands, run from repo root)

1. **Lint**: `ruff check src tests`
   Pass condition: exit code 0 (zero lint errors/warnings).

2. **Type-check**: `mypy src`
   Pass condition: exit code 0 (zero mypy errors). If the Implementor adds
   a `[tool.mypy]` section to `pyproject.toml`, that configuration is what
   `mypy src` must pass cleanly under — no `# type: ignore` used to silence
   a genuine type error without a one-line comment explaining why it's
   unavoidable.

3. **Tests + coverage**: `pytest --cov=fraud_detection --cov-report=term-missing --cov-fail-under=80`
   Pass condition: exit code 0 — all tests pass **and** line coverage on
   `src/fraud_detection` is >= 80%.

A Reviewer must run all three commands themselves and paste the actual
output into `docs/qa/QA_REPORT.md`; a criterion with no pasted command
output is a FAIL, not a PASS with a caveat.
