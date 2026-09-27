# Task estimator

Before an AI coding agent runs a task, estimate its turns, tokens, cost and time.

```bash
python -m estimator "Add rate limiting to the login endpoint, with tests" [--repo PATH]
```
```json
{"rating": {"minutes": 40, "files": 2}, "size": "small",
 "p50": {"turns": 8.5, "tokens": 300000, "cost_usd": 0.13, "minutes": 0.6},
 "p90": {"turns": 13.4, "tokens": 500000, "cost_usd": 0.21, "minutes": 0.9}}
```

## How it works

1. A small model (Haiku) rates the task with two numbers: developer minutes and files touched.
2. Turns grow with both: `turns = exp(a + b·log2(minutes) + c·log2(files + 1))`.
3. Every turn resends the conversation, so `tokens = turns·C0 + turns²/2·g`.
4. Cost and time follow from tokens and turns.

`p50` is the typical case, `p90` the worst likely case. All constants live in `params.json`.

## Layout

| File | One job |
|---|---|
| `estimator/claude.py` | The only code that calls the `claude` CLI |
| `estimator/rating.py` | Step 1: prompt, parse the reply into a `Rating(minutes, files)` |
| `estimator/model.py` | Steps 2–4: `Params`, turn and token formulas, `predict()` |
| `estimator/fitting.py` | Fit `Params` from runs, leave-one-out scoring |
| `estimator/calibrate.py` | Command: refit `params.json` from `bench/results` |
| `estimator/__main__.py` | Command: estimate one task |
| `bench/run_flows.py` | Run real Claude Code on `bench/tasks.json`, log what it spent |

## Checking it against real runs

```bash
python bench/run_flows.py --reps 2   # run Claude Code on the 12 tasks in bench/tasks.json
python -m estimator.calibrate        # compare, refit params.json
```

On 24 runs (Claude Code with claude-sonnet-5), predicting each task from the other 11:
turns 1.23× off, tokens 1.29× off, cost 1.28× off, all 12 tasks within 2×.
Two runs of the same task differ by about 1.1–1.2× on their own.

Limits: calibrated on one small repo with tasks of 4–21 turns. Estimates for big tasks
(hundreds of turns) are extrapolation until you calibrate on longer real runs.

## Tests

```bash
pip install numpy pytest
python -m pytest
```

- `tests/test_predict.py`: the maths, offline.
- `tests/test_escalation.py`: six problem areas (backend, frontend, data, devops, security, ML),
  each with an easy, medium and hard task. Estimates must rise with difficulty.
  Calls the `claude` CLI; skipped without it. The LLM rater is not deterministic, so a rare failure is possible.
