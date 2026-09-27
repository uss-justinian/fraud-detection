# Task estimator (MVP)

Give it a task description. It returns a difficulty score, the chance an agent
succeeds, and P10/P50/P90 ranges for agent turns, tokens, dollar cost and time.

```bash
pip install numpy pandas scipy scikit-learn pytest
python -m estimator "Add a CLI with products and total subcommands" --repo path/to/repo
```

Needs the `claude` CLI on PATH. It calls a small model (Haiku) once per task to fill in the rubric.

## What it computes

| Step | Formula | Source |
|---|---|---|
| 1. Rubric | An LLM rates the task on 10 ticket-only questions (clarity, hints, reasoning, scope…) plus human minutes, files touched and a turn guess | Agent Psychometrics feature set (arXiv 2604.00594, MIT) |
| 2. Difficulty | `b = ridge(rubric)`, trained on 948 SWE-bench Verified/Pro tasks, 5-fold CV r = 0.52 | Agent Psychometrics IRT difficulties |
| 3. P(success) | `σ(θ_agent − b)` | 1PL item response theory |
| 4. Turns | `log T = a0 + a1·log2(human_minutes) + a2·z(b)`, log-normal spread σ | METR: effort scales with log human time |
| 5. Tokens | `input = T·C0 + T²/2·g`, `output = T·o` | Context snowball: every turn resends the history |
| 6. Cost | `p_in·(w·cache_writes + 0.1·cache_reads) + p_out·output` | Anthropic prompt caching pricing |
| 7. Time | `T · s` | seconds per turn |

Steps 4–7 start from hand-set priors and are re-fitted from real runs by `bench/evaluate.py`,
which saves `estimator/weights/flow_params.json`.

## Checking it against real runs

```bash
python bench/run_flows.py --model claude-sonnet-5 --reps 2   # real headless Claude Code runs on bench/tasks.json
python bench/evaluate.py                                      # scores estimates, calibrates, writes bench/results/report.json
```

`bench/sandbox` is a small shopping-cart library with a deliberate bug. `bench/tasks.json`
has 12 tasks, from fixing a typo to adding stock tracking with rollback.

The evaluation reports:
- priors vs calibrated (leave one task out), as the typical factor off, share within 2×, rank correlation and P10–P90 coverage;
- baselines: global median, and the LLM guessing its own turn count;
- the noise floor: how far two runs of the same task differ;
- which signals actually track real turns;
- whether the token formula holds when the true turn count is known.

## Retraining difficulty

```bash
git clone https://github.com/dariakryvosheieva/agent-psychometrics
python scripts/train_difficulty.py agent-psychometrics
```
