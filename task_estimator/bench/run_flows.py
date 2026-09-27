"""Run real agent flows on the sample tasks and record what they actually cost.

Each run copies bench/sandbox into a fresh temp dir, runs headless Claude Code
on the task, then runs the tests. One JSON line per run goes to the output file.

    python bench/run_flows.py --model claude-sonnet-5 --reps 1 --out bench/results/actuals.jsonl
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(__file__).resolve().parent
TOOLS = "Bash Edit Write Read Glob Grep"
# Strip the parent session's identity so each run is an independent session.
DROP_ENV = ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION")


def run_one(task, rep, model, timeout):
    work = Path(tempfile.mkdtemp(prefix=f"flow_{task['id']}_"))
    shutil.copytree(BENCH / "sandbox", work, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=work)
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    cmd = ["claude", "-p", task["task"], "--model", model, "--output-format", "json",
           "--permission-mode", "acceptEdits", "--allowedTools", TOOLS]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        out = json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"id": task["id"], "rep": rep, "model": model, "error": repr(e)[:300],
                "wall_s": time.time() - t0}
    tests = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=work, capture_output=True, text=True)
    mu = out.get("modelUsage", {})
    rec = {
        "id": task["id"], "rep": rep, "model": model,
        "num_turns": out.get("num_turns"),
        "duration_s": out.get("duration_ms", 0) / 1000,
        "cost_usd": out.get("total_cost_usd"),
        "input_uncached": sum(m.get("inputTokens", 0) for m in mu.values()),
        "cache_write": sum(m.get("cacheCreationInputTokens", 0) for m in mu.values()),
        "cache_read": sum(m.get("cacheReadInputTokens", 0) for m in mu.values()),
        "output": sum(m.get("outputTokens", 0) for m in mu.values()),
        "agent_ok": out.get("subtype") == "success" and not out.get("is_error"),
        "tests_pass": tests.returncode == 0,
    }
    rec["input_total"] = rec["input_uncached"] + rec["cache_write"] + rec["cache_read"]
    shutil.rmtree(work, ignore_errors=True)
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
