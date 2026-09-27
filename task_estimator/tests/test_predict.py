"""Offline tests of the maths. No LLM calls."""

import json
import subprocess

import pytest

import estimate
from estimate import DEFAULTS, predict


@pytest.mark.parametrize("field", ["turns", "tokens", "cost_usd", "minutes"])
def test_bigger_tasks_cost_more(field):
    ladder = [predict(m, f, DEFAULTS)["p50"][field] for m, f in [(5, 1), (30, 2), (120, 4), (480, 10)]]
    assert ladder == sorted(ladder) and len(set(ladder)) == len(ladder)


def test_more_files_means_more_turns():
    assert predict(30, 8, DEFAULTS)["p50"]["turns"] > predict(30, 1, DEFAULTS)["p50"]["turns"]


def test_worst_case_above_typical():
    r = predict(60, 3, DEFAULTS)
    for field in ("turns", "tokens", "cost_usd", "minutes"):
        assert r["p90"][field] > r["p50"][field]


def test_tokens_follow_snowball_formula():
    p = {**DEFAULTS, "a": 0.0, "b": 0.0, "c": 0.0}  # forces exactly 1 turn at p50
    r = predict(1, 0, p)["p50"]
    assert r["turns"] == 1.0
    assert r["tokens"] == int(p["C0"] + p["g"] / 2 + p["out_per_turn"])


@pytest.mark.parametrize("minutes,size", [(5, "trivial"), (30, "small"), (120, "medium"), (300, "large"), (900, "very large")])
def test_size_labels(minutes, size):
    assert predict(minutes, 1, DEFAULTS)["size"] == size


def test_rate_parses_model_reply(monkeypatch):
    reply = {"result": 'Here you go:\n```json\n{"minutes": 45, "files": 3}\n```'}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(reply), ""))
    assert estimate.rate("anything") == {"minutes": 45.0, "files": 3.0}


@pytest.mark.parametrize("reply,expected", [
    ({"minutes": "30-60", "files": ["a.py", "b.py"]}, {"minutes": 45.0, "files": 2.0}),
    ({"minutes": 20, "files": "3"}, {"minutes": 20.0, "files": 3.0}),
])
def test_rate_tolerates_loose_answers(monkeypatch, reply, expected):
    out = {"result": json.dumps(reply)}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(out), ""))
    assert estimate.rate("anything") == expected
