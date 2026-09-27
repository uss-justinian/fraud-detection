"""Run real agent flows on the sample tasks and record what they actually cost.

Each run copies bench/sandbox into a fresh temp dir, runs headless Claude Code
on the task, then runs the tests. One JSON line per run goes to the output file.

    python bench/run_flows.py --model claude-sonnet-5 --reps 1 --out bench/results/actuals.jsonl
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))
from estimator import claude  # noqa: E402

AGENT_ARGS = ("--permission-mode", "acceptEdits", "--allowedTools", "Bash Edit Write Read Glob Grep")


def run_one(task, rep, model, timeout):
    """Run the agent on a fresh copy of the sandbox and record what it spent."""
    work = Path(tempfile.mkdtemp(prefix=f"flow_{task['id']}_"))
    try:
        shutil.copytree(BENCH / "sandbox", work, dirs_exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=work)
        record = {"id": task["id"], "rep": rep, "model": model}
        try:
            out = claude.run(task["task"], model=model, cwd=work, extra_args=AGENT_ARGS, timeout=timeout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            return {**record, "error": repr(e)[:300]}
        tests = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=work, capture_output=True, text=True)
        return {**record, **usage(out), "tests_pass": tests.returncode == 0}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def usage(out):
    """Turns, time, cost and tokens from a `claude -p --output-format json` result."""
    per_model = out.get("modelUsage", {}).values()

    def total(key):
        return sum(m.get(key, 0) for m in per_model)

    rec = {"num_turns": out.get("num_turns"), "duration_s": out.get("duration_ms", 0) / 1000,
           "cost_usd": out.get("total_cost_usd"),
           "input_uncached": total("inputTokens"), "cache_write": total("cacheCreationInputTokens"),
           "cache_read": total("cacheReadInputTokens"), "output": total("outputTokens"),
           "agent_ok": out.get("subtype") == "success" and not out.get("is_error")}
    rec["input_total"] = rec["input_uncached"] + rec["cache_write"] + rec["cache_read"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default=str(BENCH / "results" / "actuals.jsonl"))
    a = ap.parse_args()
    tasks = json.loads((BENCH / "tasks.json").read_text())
    if a.only:
        tasks = [t for t in tasks if t["id"] in a.only]
    jobs = [(t, r) for r in range(a.reps) for t in tasks]
    with ThreadPoolExecutor(a.parallel) as ex, open(a.out, "a") as f:
        for rec in ex.map(lambda j: run_one(j[0], j[1], a.model, a.timeout), jobs):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["id"], rec["rep"], rec.get("num_turns"), rec.get("input_total"), rec.get("cost_usd"), flush=True)


if __name__ == "__main__":
    main()
