"""Turn rubric scores into difficulty, P(success), turns, tokens, cost and time.

Pipeline (see README for the sources behind each step):
  1. difficulty  = ridge(rubric)                      Agent Psychometrics
  2. P(success)  = sigmoid(theta_agent - difficulty)  1PL IRT
  3. log T       = a0 + a1*log2(human_minutes) + a2*z(difficulty)  (METR: effort ~ log human time)
  4. input       = T*C0 + T^2/2 * g                   context snowball: every turn resends the history
     output      = T*o
  5. cost        = p_in*(w*cache_writes + 0.1*cache_reads + uncached) + p_out*output
  6. time        = T * s
Log-normal spread on T gives the P10/P50/P90 ranges.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent / "weights"
Z10, Z90 = -1.2816, 1.2816


@dataclass
class Params:
    """Everything calibrate.py re-fits from logged runs. Defaults are hand-set priors."""
    a0: float = math.log(5.0) - 0.45 * math.log2(5)  # ~5 turns for a 5-minute task
    a1: float = 0.45                                  # x1.57 turns per doubling of human time
    a2: float = 0.10                                  # per standard deviation of difficulty
    sigma: float = 0.5                                # log-space spread of T
    C0: float = 42000.0      # starting context: system prompt, tools, task (measured on Claude Code)
    g: float = 1500.0        # tokens added to the context per turn (output + tool results)
    o: float = 300.0         # output tokens per turn
    s: float = 6.0           # seconds per turn
    theta: float = 3.1       # agent ability on the SWE-bench Verified IRT scale (top agents, late 2025)
    p_in: float = 2.0        # $ per million input tokens
    p_out: float = 10.0      # $ per million output tokens
    write_mult: float = 2.0  # cache write price multiple (1-hour cache)
    c0_write_frac: float = 1.0  # share of C0 written fresh; ~0 when the harness prompt is already cached
    source: str = "priors"
    fit_stats: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=WEIGHTS / "flow_params.json"):
        return cls(**json.loads(Path(path).read_text())) if Path(path).exists() else cls()

    def save(self, path=WEIGHTS / "flow_params.json"):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def difficulty(scores):
    w = json.loads((WEIGHTS / "difficulty.json").read_text())
    b = w["intercept"] + sum(w["coef"][f] * (scores[f] - w["mean"][f]) / w["scale"][f] for f in w["features"])
    return b, (b - w["difficulty_mean"]) / w["difficulty_std"]


def log_turns_mean(scores, z, p):
    return p.a0 + p.a1 * math.log2(max(scores["human_minutes"], 1.0)) + p.a2 * z


def tokens_for_turns(T, p):
    inp = T * p.C0 + T * T / 2 * p.g
    writes = min(inp, p.c0_write_frac * p.C0 + T * p.g)  # each new context chunk is written to cache once
    return {"input_total": inp, "cache_write": writes, "cache_read": inp - writes, "output": T * p.o}


def cost(tok, p):
    return (p.p_in * (p.write_mult * tok["cache_write"] + 0.1 * tok["cache_read"]) + p.p_out * tok["output"]) / 1e6


def estimate(scores, p=None):
    p = p or Params.load()
    b, z = difficulty(scores)
    mu = log_turns_mean(scores, z, p)
    out = {"difficulty": round(b, 2), "difficulty_z": round(z, 2),
           "p_success": round(1 / (1 + math.exp(-(p.theta - b))), 2),
           "human_minutes": scores["human_minutes"]}
    for q, zq in (("p10", Z10), ("p50", 0.0), ("p90", Z90)):
        T = math.exp(mu + zq * p.sigma)
        tok = tokens_for_turns(T, p)
        out[q] = {"turns": round(T, 1), "input_tokens": int(tok["input_total"]),
                  "output_tokens": int(tok["output"]), "cost_usd": round(cost(tok, p), 3),
                  "minutes": round(T * p.s / 60, 1)}
    out["expected_cost_with_retries"] = round(out["p50"]["cost_usd"] / max(out["p_success"], 0.05), 3)
    out["complexity"] = ("trivial", "small", "medium", "large", "very large")[
        sum(scores["human_minutes"] > m for m in (10, 45, 180, 600))]
    out["params"] = p.source
    return out
