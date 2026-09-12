# QA Report: Fraud Detection ML Pipeline

Reviewer: QA agent, following `docs/process/reviewer.md`.
Reviewed commit: `c23a993` ("Add tests for fraud-detection pipeline per QA_PLAN.md"), branch `claude/fraud-detection-pipeline-nn8ijr`.

Verdict: **APPROVED**

All three quality gates pass with real, independently-run output; every
acceptance criterion in `QA_PLAN.md` Section 1 is satisfied with concrete
evidence; no scope creep or spec violations were found.

---

## 1. Quality Gates (run from repo root, fresh virtualenv)

Setup:

```
$ python3 -m venv /tmp/qa_venv
$ /tmp/qa_venv/bin/pip install --upgrade pip
$ /tmp/qa_venv/bin/pip install -e ".[dev]"
```

(installed cleanly, no errors)

### 1.1 Lint — `ruff check src tests`

```
All checks passed!
EXIT CODE: 0
```

**PASS.**

### 1.2 Type-check — `mypy src`

```
Success: no issues found in 8 source files
EXIT CODE: 0
```

**PASS.** No `# type: ignore` used anywhere in `src/`.

### 1.3 Tests + coverage — `pytest --cov=fraud_detection --cov-report=term-missing --cov-fail-under=80`

```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/user/fraud-detection
configfile: pyproject.toml
plugins: cov-7.1.0
collected 53 items

tests/test_cli.py ............                                           [ 22%]
tests/test_config.py ....                                                [ 30%]
tests/test_data.py ........                                              [ 45%]
tests/test_evaluate.py ......                                            [ 56%]
tests/test_features.py ......                                            [ 67%]
tests/test_model.py .......                                              [ 81%]
tests/test_pipeline.py ..........                                        [100%]

=============================== warnings summary ===============================
tests/test_cli.py: 12 warnings
  .../sklearn/metrics/_ranking.py:1192: UserWarning: No positive class found in y_true, recall is set to one for all thresholds.
    warnings.warn(

tests/test_cli.py::test_train_then_evaluate_round_trip (x2)
tests/test_cli.py::test_train_both_model_types_succeed_no_code_change (x2)
tests/test_cli.py::test_evaluate_prints_metrics (x2)
  .../sklearn/metrics/_ranking.py:469: UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
    warnings.warn(

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.11.15-final-0 _______________

Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/fraud_detection/__init__.py       1      0   100%
src/fraud_detection/cli.py           65      1    98%   128
src/fraud_detection/config.py        17      0   100%
src/fraud_detection/data.py          92      0   100%
src/fraud_detection/evaluate.py      38      0   100%
src/fraud_detection/features.py      23      0   100%
src/fraud_detection/model.py         16      0   100%
src/fraud_detection/pipeline.py      60      0   100%
---------------------------------------------------------------
TOTAL                               312      1    99%
Required test coverage of 80% reached. Total coverage: 99.68%
======================= 53 passed, 18 warnings in 8.89s ========================
EXIT CODE: 0
```

**PASS.** 53/53 tests pass; 99.68% line coverage on `src/fraud_detection`
(threshold 80%). The warnings are sklearn `UndefinedMetricWarning`/
`UserWarning` from CLI round-trip tests that use small (500-row, ~1.5% fraud
rate) generated datasets where one class can be entirely absent from a
tiny test split — cosmetic, not a test failure, and not a code defect.

---

## 2. Acceptance Criteria (`QA_PLAN.md` Section 1)

### Data generation (`data.py`)

| Criterion | Verdict | Evidence |
|---|---|---|
| Returns `n_samples` rows with the raw schema columns | PASS | `tests/test_data.py::test_returns_expected_row_count_and_schema`; independently confirmed via `generate-data --n-samples 500` CLI run producing a 500-row CSV. |
| Fixed seed → byte-identical DataFrames | PASS | `tests/test_data.py::test_reproducible_for_fixed_seed` (`pd.testing.assert_frame_equal`). Code review: `np.random.default_rng(config.random_seed)` is constructed fresh inside `generate_transactions` on every call — no shared/global RNG state. |
| `timestamp` non-decreasing | PASS | `tests/test_data.py::test_timestamp_is_non_decreasing`; also enforced structurally by `data.py:78` (`sort_values("timestamp", ...)` before return). |
| Realized fraud rate within ±30% relative, n≥5000 | PASS | `tests/test_data.py::test_realized_fraud_rate_within_tolerance` (n=5000, fraud_rate=0.02). |
| `is_fraud` boolean, no nulls | PASS | `tests/test_data.py::test_is_fraud_is_boolean_with_no_nulls`. |

Spot check beyond the checklist: read `data.py` in full. The 3 fraud
sub-populations are implemented with the spec's fixed 40/40/20 proportions
(`_FRAUD_FOREIGN_HIGH_VALUE_FRAC`, `_FRAUD_VELOCITY_BURST_FRAC`, remainder),
not exposed via `Config`. Pattern 1 (foreign+high-value+new-account),
pattern 2 (velocity burst, hour forced into `[0,5]`, burst window of
minutes, elevated distance), and pattern 3 (noise, drawn from
`_generate_base` with `is_fraud_label=True`, i.e. literally the same
generator as legitimate rows) all match SPEC Section 4. The ~2% legitimate
"business travel" overlap (`_LEGIT_BUSINESS_TRAVEL_FRAC`) is applied only to
legitimate rows, not to fraud noise rows, matching spec intent. Customer
pool size (`max(200, n_samples // 20)`) matches spec formula and is
independently tested (`test_customer_pool_size_matches_spec_formula`).

### Feature engineering (`features.py`)

| Criterion | Verdict | Evidence |
|---|---|---|
| `add_features` does not mutate input | PASS | `tests/test_features.py::test_does_not_mutate_input`; code does `df.copy(deep=True)` at `features.py:24`. |
| `transactions_last_24h` = `[0, 1, 0]` on the 3-row fixture | PASS | `tests/test_features.py::test_transactions_last_24h_exact_fixture`. I independently re-derived this by hand-running `add_features` on the exact fixture plus two additional cases (see below) — matches. |
| `amount_log`/`hour_of_day`/`is_weekend`/`is_night` match hand-computed values | PASS | `tests/test_features.py::test_amount_log_hour_weekend_night_fixture` and `test_is_night_boundary_inclusive_of_hour_5` (confirms hour 5 is night, hour 6 is not — matches SPEC's "`[0,5]` inclusive"). |
| Output contains every `Config.numeric_features`/`categorical_features` column | PASS | `tests/test_features.py::test_output_contains_all_configured_feature_columns`. |

**Independent spot-check of the leakage-critical `transactions_last_24h` logic**
(SPEC's explicit "verify yourself" item): I ran `add_features` manually
(not from the test suite) on:
- The exact spec fixture (t0, t0+1h, t0+25h) → got `[0, 1, 0]`, matching.
- A boundary case: two transactions for the same customer exactly 24h apart
  → got `[0, 0]` (neither counts the other), confirming the window is
  strictly `(t-24h, t)`, i.e. exactly-24h-prior is correctly excluded, not
  off-by-one included.
- The same fixture with rows presented to `add_features` in reverse
  (non-chronological) input order → identical per-transaction results,
  confirming the implementation's internal per-customer sort
  (`features.py:47`, `np.argsort(... kind="mergesort")`) makes the result
  independent of input row order, not just of already-sorted input.

Code review of `_transactions_last_24h` (`features.py:33-61`): uses
`searchsorted` with `side="right"` for the window's lower bound and
`side="left"` for the row's own timestamp against itself, which correctly
implements "strictly earlier than the row, and strictly after `t-24h`" with
no off-by-one and no reliance on unique timestamps. Grouping is per
`customer_id`, satisfying the "excludes other customers" requirement
(also directly tested in `test_transactions_last_24h_excludes_other_customers`).

### Split (`pipeline.py`)

| Criterion | Verdict | Evidence |
|---|---|---|
| Train/test sizes match `round(len*(1-test_size))` ±1 | PASS | `tests/test_pipeline.py::test_time_split_sizes_and_ordering` (parametrized 0.2/0.25/0.3). |
| `max(train.timestamp) <= min(test.timestamp)` | PASS | Same test, asserts `train["timestamp"].max() <= test["timestamp"].min()`. |
| No row duplicated/dropped | PASS | `tests/test_pipeline.py::test_time_split_no_row_dropped_or_duplicated`. |

Spot check: read `time_split` (`pipeline.py:23-36`) directly — it sorts by
timestamp ascending (stable mergesort), takes a contiguous `iloc` prefix/
suffix split by position, so no row can appear in both partitions or be
dropped by construction, independent of the test. This is a positional
split on chronologically sorted data, not a random shuffle — satisfies
SPEC Section 6's leakage-avoidance rationale directly (no test transaction's
timestamp can precede a train transaction's timestamp, apart from
legitimate ties at the boundary, which the QA_PLAN's `<=` bound explicitly
allows for).

### Model (`model.py`)

| Criterion | Verdict | Evidence |
|---|---|---|
| `build_pipeline` returns unfitted `Pipeline` | PASS | `tests/test_model.py::test_build_pipeline_returns_unfitted_pipeline` (both model types, uses `check_is_fitted` + `NotFittedError`). |
| `class_weight == "balanced"` for both model types | PASS | `tests/test_model.py::test_class_weight_is_balanced`; also directly visible in `model.py:39,45`. |
| Invalid `model_type` raises `ValueError` | PASS | `tests/test_model.py::test_invalid_model_type_raises_value_error`. |
| Both types `.fit()`/`.predict_proba()` without raising | PASS | `tests/test_model.py::test_pipeline_fits_and_predicts_proba`, built from real `generate_transactions`+`add_features` output. |

Code review confirms no model-type branching exists outside `model.py`
(`pipeline.py` and `cli.py` only pass `config.model_type` through), matching
SPEC Section 8's "config change only" requirement.

### Evaluation (`evaluate.py`)

| Criterion | Verdict | Evidence |
|---|---|---|
| `pr_auc`/`roc_auc` match sklearn directly | PASS | `tests/test_evaluate.py::test_pr_auc_and_roc_auc_match_sklearn_directly`. |
| `recall_at_precision`/`threshold` match manual derivation | PASS | `tests/test_evaluate.py::test_recall_at_precision_matches_manual_derivation`, independently re-derives via `precision_recall_curve` in the test itself. |
| Unreachable target precision → `threshold=None`, `recall_at_precision=None`, no raise | PASS | `tests/test_evaluate.py::test_unreachable_target_precision_returns_none_without_raising` (all-identical `y_prob`, target 0.99). |
| `confusion_matrix` keys exactly `{tn,fp,fn,tp}`, sums to `len(y_true)` | PASS | `tests/test_evaluate.py::test_confusion_matrix_keys_and_total` and `test_confusion_matrix_stable_shape_when_threshold_unreachable` (also confirms the "effective threshold 1.0" always-negative fallback). |
| `write_model_card` contains "synthetic", `model_type`, `pr_auc` value | PASS | `tests/test_evaluate.py::test_write_model_card_contains_required_content`. I additionally read the full generated model card from a live `train` run (see CLI section below) — it also contains the required Data Provenance disclaimer, Config snapshot, Metrics, Intended Use, and Limitations sections per SPEC Section 10. |

### CLI / pipeline integration

All of the following were re-run independently by me from a clean
`/tmp/qa_e2e` directory using the fresh `/tmp/qa_venv` interpreter (not the
Implementor's reported output):

| Criterion | Verdict | Evidence |
|---|---|---|
| `generate-data --n-samples 500 --seed 1 --output <tmp>/t.csv` exits 0, writes 500-row CSV | PASS | Ran directly: `Wrote 2000 rows to /tmp/qa_e2e/t.csv` (I used 2000 for the e2e bound check below; 500-row case is also covered by `tests/test_cli.py::test_generate_data_writes_csv` and I confirmed the same command shape works). |
| `train --model-type logistic_regression` exits 0, produces a loadable model + model card | PASS | Ran directly; `joblib.load` succeeded; `models/model_card.md` (relative to cwd, per `config.model_card_path` default) was written and contains the expected sections. |
| `train --model-type random_forest` also succeeds, no code change | PASS | Ran directly on the same data file: exited 0, produced a loadable model. |
| `evaluate` exits 0, prints `pr_auc`, `roc_auc`, confusion matrix | PASS | Ran directly: `[evaluate] pr_auc=... roc_auc=...` and `[evaluate] confusion_matrix={...}` printed to stdout. |
| `train --model-type not-a-model` exits non-zero, names valid choices, no stack trace | PASS | Ran directly: exit code 2, stderr: `fraud-detection train: error: argument --model-type: invalid choice: 'not-a-model' (choose from 'logistic_regression', 'random_forest')` — no traceback. (argparse `choices=` rejects this before it would ever reach `model.py`'s `ValueError` branch; the `ValueError` path is still real and tested for direct `Config`/`build_pipeline` use in `tests/test_model.py`.) |
| `--help` and each subcommand `--help` exit 0 | PASS | Ran `fraud_detection.cli --help` and `generate-data --help` directly: both exit 0 with usage text. `train --help`/`evaluate --help` covered by `tests/test_cli.py::test_subcommand_help_exits_zero` (parametrized). |
| End-to-end 2000-row: `fraud_rate < pr_auc < 0.999` | PASS | Ran directly, seed 42 (Config default), logistic_regression: `pr_auc=0.5003` (> fraud_rate 0.015, < 0.999). Random forest on the same data: `pr_auc=0.8435` (also within bounds). Also covered by `tests/test_pipeline.py::test_run_train_end_to_end` (both model types, seed 13, n=2000). |
| Missing data/model file → non-zero exit, clear message | PASS | Ran directly: `train --data-path <missing>` → `Error: Data file not found: ...`, exit 1. `evaluate --model-path <missing>` → `Error: Model file not found: ...`, exit 1. Both clean error messages, no traceback. |

### Non-scope / dependency guardrails

| Criterion | Verdict | Evidence |
|---|---|---|
| Runtime deps unchanged (pandas, numpy, scikit-learn, joblib only) | PASS | `pyproject.toml` `[project].dependencies` — confirmed unchanged from spec list; `git show d79888b -- pyproject.toml` diff shows only `[project.optional-dependencies].dev`, `[project.scripts]`, and `[tool.mypy]` were added, not `[project].dependencies`. |
| No `smote`/`imblearn`/`SMOTE` in `src/` | PASS | `grep -ri "smote\|imblearn" src -r` → no matches. |
| No `predict` subcommand | PASS | `grep -n "predict" src/fraud_detection/cli.py` → no matches; `_build_parser` only registers `generate-data`, `train`, `evaluate`. Also directly tested: `tests/test_cli.py::test_no_predict_subcommand_exists` runs `cli.main(["predict", ...])` and asserts `SystemExit` (argparse rejects the unknown subcommand). |

**All acceptance criteria: PASS. No FAILs.**

---

## 3. Defects / Observations Beyond the Checklist

No blocking defects found. Two non-blocking observations:

1. **Model card's `n_samples`/`fraud_rate` reflect the `Config` object's
   fields, not the actual size/composition of the CSV file that was
   trained on, when `--data-path` points at externally-generated data.**
   Concretely: run `generate-data --n-samples 100 --output x.csv` (default
   `Config(n_samples=20000)` is otherwise unused for generation-time
   sizing since `--n-samples` overrides it) then
   `train --data-path x.csv` with no `--seed`/other override — the model is
   trained on the real 100-row file, but `run_train` passes the CLI's own
   fresh `Config()` (`n_samples=20000`, the dataclass default) into
   `write_model_card`, which prints `n_samples: 20000` in the "Configuration"
   section even though only 100 rows were actually used. This is because
   `train` has no `--n-samples` flag (correctly, per SPEC Section 11 — data
   generation and training are separate steps) and the model card documents
   the `Config` snapshot passed into `run_train`, not runtime facts about
   the loaded file. SPEC Section 10 literally asks for "Config snapshot:
   `random_seed`, `n_samples`, ... " (the dataclass fields), so this is
   arguably spec-compliant as literally written, but it can produce a
   misleading model card in the (very plausible) case where someone trains
   on a differently-sized dataset than the default `Config`. Not a test
   failure and not a blocking defect — flagging as a follow-up: the model
   card would be more useful documenting the actual training-row count
   (e.g. `len(train_df)`) alongside/instead of `config.n_samples`.

2. **Cosmetic test-run warnings**: several `test_cli.py` round-trip tests
   use very small (500-row, ~1.5% fraud rate) generated datasets, which can
   yield a test split with only one class present, triggering sklearn's
   `UndefinedMetricWarning`/`UserWarning` (visible in the pytest output
   above). Tests still pass and assert the right things; this is not a
   correctness bug, just non-silent test output. No action required.

Neither observation blocks approval.

---

## 4. Verdict

**APPROVED.**

- All three quality gates (`ruff check src tests`, `mypy src`,
  `pytest --cov=fraud_detection --cov-report=term-missing --cov-fail-under=80`)
  pass with real, independently-reproduced output (Section 1 above).
- Every acceptance criterion in `QA_PLAN.md` Section 1 is PASS, backed by a
  specific test, a command I ran myself, or code I read directly (Section 2
  above) — including independent manual verification of the two
  spec-critical/subtle behaviors called out in the review brief: the
  backward-only-window `transactions_last_24h` logic (including an
  exact-24h boundary case and row-order independence, beyond what the test
  suite covers) and the invalid `--model-type` CLI path. The leakage-free
  nature of `time_split` was confirmed by direct code reading plus the
  existing test evidence.
- No scope creep: no new runtime dependencies, no resampling code, no
  `predict` command, all confirmed via `grep` and direct CLI/argparse
  inspection.
- Two minor, non-blocking observations noted above for optional follow-up.
