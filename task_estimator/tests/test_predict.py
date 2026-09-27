"""Offline tests of the maths and parsing. No LLM calls."""

import json
import subprocess

import pytest

from estimator import claude
from estimator.fitting import fit
from estimator.model import Params, predict
from estimator.rating import Rating, parse, rate

P = Params(a=0.862, b=0.136, c=0.475, sigma=0.35, C0=31210, g=950, out_per_turn=459, sec_per_turn=3.9,
           k_in=0.283, k_out=12.809)
FIELDS = ("turns", "tokens", "cost_usd", "minutes")


@pytest.mark.parametrize("field", FIELDS)
def test_bigger_tasks_cost_more(field):
    ladder = [predict(Rating(m, f), P)["p50"][field] for m, f in [(5, 1), (30, 2), (120, 4), (480, 10)]]
    assert ladder == sorted(ladder) and len(set(ladder)) == len(ladder)


def test_more_files_means_more_turns():
    assert predict(Rating(30, 8), P)["p50"]["turns"] > predict(Rating(30, 1), P)["p50"]["turns"]


def test_worst_case_above_typical():
    r = predict(Rating(60, 3), P)
    assert all(r["p90"][f] > r["p50"][f] for f in FIELDS)


def test_tokens_follow_snowball_formula():
    one_turn = Params(**{**vars(P), "a": 0.0, "b": 0.0, "c": 0.0})
    r = predict(Rating(1, 0), one_turn)["p50"]
    assert r["turns"] == 1.0
    assert r["tokens"] == int(P.C0 + P.g / 2 + P.out_per_turn)


@pytest.mark.parametrize("minutes,size", [(5, "trivial"), (30, "small"), (120, "medium"), (300, "large"), (900, "very large")])
def test_size_labels(minutes, size):
    assert predict(Rating(minutes, 1), P)["size"] == size


@pytest.mark.parametrize("reply,expected", [
    ('Here you go:\n```json\n{"minutes": 45, "files": 3}\n```', Rating(45, 3)),
    ('{"minutes": "30-60", "files": ["a.py", "b.py"]}', Rating(45, 2)),
    ('{"minutes": 20, "files": "3"}', Rating(20, 3)),
])
def test_parse_tolerates_loose_answers(reply, expected):
    assert parse(reply) == expected


def test_parse_rejects_reply_without_numbers():
    with pytest.raises(ValueError):
        parse("I'll look at the repository first.")


def test_rate_uses_injected_model():
    prompts = []
    r = rate("Fix the typo", ask=lambda p: prompts.append(p) or '{"minutes": 5, "files": 1}')
    assert r == Rating(5, 1) and "Fix the typo" in prompts[0]


def test_claude_run_drops_parent_session(monkeypatch):
    seen = {}
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, env=kw["env"])
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": "ok"}), "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert claude.run("hi", model="m", cwd=".")["result"] == "ok"
    assert "CLAUDE_CODE_SESSION_ID" not in seen["env"] and seen["cmd"][:3] == ["claude", "-p", "hi"]


def test_fit_recovers_known_params():
    ratings = {f"t{i}": Rating(m, f) for i, (m, f) in enumerate([(5, 1), (20, 2), (60, 3), (240, 6), (30, 1)])}
    runs = []
    for task_id, rating in ratings.items():
        est = predict(rating, P)["p50"]
        t = est["turns"]
        tin = P.C0 * t + P.g * t * t / 2
        runs.append({"id": task_id, "num_turns": t, "input_total": tin, "output": P.out_per_turn * t,
                     "cost_usd": (P.k_in * tin + P.k_out * P.out_per_turn * t) / 1e6, "duration_s": P.sec_per_turn * t})
    got = fit(runs, ratings)
    assert got.C0 == pytest.approx(P.C0, rel=0.02) and got.g == pytest.approx(P.g, rel=0.05)
    assert got.c == pytest.approx(P.c, abs=0.05)
