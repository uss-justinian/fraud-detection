"""Ask a cheap LLM to rate a task on the rubric.

The first ten questions are the ticket-only features from Agent Psychometrics
(arXiv 2604.00594, MIT licence); the difficulty model was trained on them. The
last three are ours and feed the turn and time estimates.
"""

import json
import os
import re
import subprocess

RUBRIC = """\
- solution_hint (0-3): does the task text hint at the solution? 0 none, 1 vague direction, 2 clear approach, 3 exact steps.
- domain_knowledge_required (1-5): 1 basic language use, 2 standard library, 3 framework-specific, 4 deep library internals, 5 obscure/specialised.
- logical_reasoning_required (1-5): 1 mechanical, 2 simple cause-effect, 3 multi-step, 4 complex multi-factor, 5 deep edge cases/invariants.
- atypicality (1-5): 1 very common pattern ... 5 rare or novel.
- verification_difficulty (1-5): 1 trivial pass/fail, 2 straightforward tests, 3 some edge cases, 4 hard, 5 very hard to verify.
- error_specificity (1-5): 1 vague symptoms ... 5 exact error with stack trace (for non-bug tasks: how precisely the target is specified).
- debugging_complexity (1-5): 1 obvious cause ... 5 deep investigation needed.
- codebase_scope (1-5): 1 single file ... 5 system-wide.
- similar_issue_likelihood (0 or 1): 1 if this is a common task type with known solutions.
- side_effect_risk (1-5): 1 none ... 5 critical risk of breaking other things.
- human_minutes (number): minutes a competent developer new to this codebase would need, including testing.
- files_touched (integer): files likely created or modified.
- agent_turns_guess (integer): your guess of how many tool calls a coding agent would make."""

PROMPT = """You are rating a software task before anyone works on it. Rate it on each item below.

{rubric}

{context}TASK:
{task}

Reply with only a JSON object whose keys are exactly the item names above."""

KEYS = [line.split(" ")[1] for line in RUBRIC.splitlines()]
DROP_ENV = ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION")


def repo_context(path):
    """A short file tree with line counts, so the rater can judge scope."""
    lines = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                n = sum(1 for _ in open(p, errors="ignore"))
            except OSError:
                continue
            lines.append(f"{os.path.relpath(p, path)} ({n} lines)")
    return "REPOSITORY FILES:\n" + "\n".join(lines[:200]) + "\n\n"


def rate(task, repo=None, model="claude-haiku-4-5-20251001"):
    prompt = PROMPT.format(rubric=RUBRIC, context=repo_context(repo) if repo else "", task=task)
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    p = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format", "json",
                        "--tools", ""], capture_output=True, text=True, env=env, timeout=180)
    text = json.loads(p.stdout)["result"]
    scores = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    missing = [k for k in KEYS if k not in scores]
    if missing:
        raise ValueError(f"rater omitted {missing}")
    return {k: float(scores[k]) for k in KEYS}
