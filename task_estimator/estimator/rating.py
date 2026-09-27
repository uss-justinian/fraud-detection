"""Step 1: a small model rates the task with two numbers."""

import json
import os
import re
import tempfile
from dataclasses import dataclass

from estimator import claude

RATER_MODEL = "claude-haiku-4-5-20251001"
SYSTEM = "You estimate software tasks. You have no tools and cannot see any code. Reply with only the JSON asked for."
PROMPT = """Rate this software task before anyone works on it.
Reply with only JSON of two numbers: {{"minutes": <minutes a competent developer new to the codebase needs, including testing>, "files": <count of files likely created or modified>}}
{context}
TASK:
{task}"""


@dataclass(frozen=True)
class Rating:
    minutes: float
    files: float


def rate(task, repo=None, ask=None):
    """Rate a task. `ask` maps a prompt to reply text; defaults to Haiku via the Claude CLI."""
    return parse((ask or _ask_claude)(build_prompt(task, repo)))


def build_prompt(task, repo=None):
    return PROMPT.format(context=_repo_files(repo) if repo else "", task=task)


def parse(reply):
    """Pull minutes and files out of a loosely formatted reply."""
    values = {}
    for key in ("minutes", "files"):
        m = re.search(rf'"{key}"\s*:\s*(\[.*?\]|"[^"]*"|[\d.]+)', reply, re.S)
        if not m:
            raise ValueError(f"no {key!r} in rater reply: {reply[:200]!r}")
        values[key] = _number(m.group(1))
    return Rating(**values)


def _number(raw):
    """A number, a list of file names (counted) or a range like "30-60" (averaged)."""
    if raw.startswith("["):
        return float(len(json.loads(raw)))
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
    return sum(nums) / len(nums)


def _ask_claude(prompt):
    with tempfile.TemporaryDirectory() as empty:  # no project files or CLAUDE.md to distract the rater
        return claude.run(prompt, model=RATER_MODEL, cwd=empty,
                          extra_args=("--tools", "", "--system-prompt", SYSTEM))["result"]


def _repo_files(path, limit=200):
    names = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        names += [os.path.relpath(os.path.join(root, f), path) for f in files]
    return "\nREPOSITORY FILES:\n" + "\n".join(sorted(names)[:limit]) + "\n"
