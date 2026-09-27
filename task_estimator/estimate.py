"""Estimate what an AI coding agent will spend on a task, before it runs.

    python estimate.py "Add a CLI with products and total subcommands" [--repo PATH]

1. A small LLM rates the task: developer minutes, files touched.
2. turns  = exp(a + b*log2(minutes) + c*log2(files + 1))
3. tokens = turns*C0 + turns^2/2 * g   (each turn resends the whole conversation)
4. cost   = k_in*input + k_out*output,  time = turns * seconds_per_turn
Constants are fitted from real runs by calibrate.py and stored in params.json.
"""

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent / "params.json"
DEFAULTS = {  # fitted on 24 Claude Code (claude-sonnet-5) runs, see bench/
    "a": 0.862, "b": 0.136, "c": 0.475, "sigma": 0.35,
    "C0": 31210, "g": 950, "out_per_turn": 459, "sec_per_turn": 3.9,
    "k_in": 0.283, "k_out": 12.809,  # effective $ per million tokens, caching included
}
Z90 = 1.2816
DROP_ENV = ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION")

SYSTEM = "You estimate software tasks. You have no tools and cannot see any code. Reply with only the JSON asked for."
PROMPT = """Rate this software task before anyone works on it.
Reply with only JSON of two numbers: {{"minutes": <minutes a competent developer new to the codebase needs, including testing>, "files": <count of files likely created or modified>}}
{context}
TASK:
{task}"""


def params():
    return {**DEFAULTS, **json.loads(PARAMS_FILE.read_text())} if PARAMS_FILE.exists() else DEFAULTS


def repo_files(path):
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        out += [os.path.relpath(os.path.join(root, f), path) for f in files]
    return "\nREPOSITORY FILES:\n" + "\n".join(sorted(out)[:200]) + "\n"


def rate(task, repo=None, model="claude-haiku-4-5-20251001"):
    """Ask a small model for {"minutes", "files"}."""
    prompt = PROMPT.format(context=repo_files(repo) if repo else "", task=task)
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    with tempfile.TemporaryDirectory() as empty:  # no project files or CLAUDE.md to distract the rater
        p = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format", "json", "--tools", "",
                            "--system-prompt", SYSTEM], cwd=empty, capture_output=True, text=True, env=env, timeout=180)
    reply = json.loads(p.stdout)["result"]
    return {k: _number(re.search(rf'"{k}"\s*:\s*(\[.*?\]|"[^"]*"|[\d.]+)', reply, re.S).group(1))
            for k in ("minutes", "files")}


def _number(raw):
    """A number, a list of file names (counted) or a range like "30-60" (averaged)."""
    if raw.startswith("["):
        return float(len(json.loads(raw)))
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
    return sum(nums) / len(nums)


def predict(minutes, files, p=None):
    """Typical (p50) and worst-case (p90) turns, tokens, cost and time."""
    p = p or params()
    mu = p["a"] + p["b"] * math.log2(max(minutes, 1)) + p["c"] * math.log2(max(files, 0) + 1)
    out = {"size": ("trivial", "small", "medium", "large", "very large")[sum(minutes > m for m in (10, 45, 180, 600))]}
    for q, z in (("p50", 0.0), ("p90", Z90)):
        turns = math.exp(mu + z * p["sigma"])
        tokens_in = turns * p["C0"] + turns ** 2 / 2 * p["g"]
        tokens_out = turns * p["out_per_turn"]
        out[q] = {"turns": round(turns, 1), "tokens": int(tokens_in + tokens_out),
                  "cost_usd": round((p["k_in"] * tokens_in + p["k_out"] * tokens_out) / 1e6, 3),
                  "minutes": round(turns * p["sec_per_turn"] / 60, 1)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--repo")
    a = ap.parse_args()
    r = rate(a.task, a.repo)
    print(json.dumps({"rating": r, **predict(r["minutes"], r["files"])}, indent=2))
