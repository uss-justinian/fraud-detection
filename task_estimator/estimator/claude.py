"""The one place that shells out to the Claude Code CLI."""

import json
import os
import subprocess

# Unset so a child run is its own session, not part of the one that launched it.
_PARENT_SESSION_VARS = ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDECODE",
                        "CLAUDE_CODE_CHILD_SESSION")


def run(prompt, *, model, cwd, extra_args=(), timeout=180):
    """Run `claude -p` headless and return its JSON result (usage, turns, cost, reply text)."""
    env = {k: v for k, v in os.environ.items() if k not in _PARENT_SESSION_VARS}
    p = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format", "json", *extra_args],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return json.loads(p.stdout)
