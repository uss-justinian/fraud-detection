"""Steps 2-4: rating -> turns -> tokens, cost and time.

    turns  = exp(a + b*log2(minutes) + c*log2(files + 1))
    tokens = turns*C0 + turns^2/2 * g      (each turn resends the whole conversation)
    cost   = k_in*input + k_out*output,  time = turns * sec_per_turn
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent.parent / "params.json"
QUANTILES = {"p50": 0.0, "p90": 1.2816}  # standard-normal z for each reported quantile
SIZES = ((10, "trivial"), (45, "small"), (180, "medium"), (600, "large"), (math.inf, "very large"))


@dataclass(frozen=True)
class Params:
    a: float             # turn model: intercept
    b: float             #   per doubling of developer minutes
    c: float             #   per doubling of (files + 1)
    sigma: float         # log-space spread of turns
    C0: float            # starting context tokens
    g: float             # context tokens added per turn
    out_per_turn: float  # output tokens per turn
    sec_per_turn: float
    k_in: float          # effective $ per million input tokens, caching included
    k_out: float         # $ per million output tokens

    @classmethod
    def load(cls, path=PARAMS_FILE):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path=PARAMS_FILE):
        Path(path).write_text(json.dumps({k: round(v, 4) for k, v in asdict(self).items()}, indent=2))


def turn_features(rating):
    """Inputs of the log-linear turn model, shared by prediction and fitting."""
    return (1.0, math.log2(max(rating.minutes, 1)), math.log2(max(rating.files, 0) + 1))


def token_basis(turns):
    """input tokens = C0*turns + g*turns^2/2; shared by prediction and fitting."""
    return (turns, turns * turns / 2)


def size(minutes):
    return next(label for limit, label in SIZES if minutes <= limit)


def at_turns(turns, p):
    """Tokens, cost and time for a given number of turns."""
    tokens_in = sum(x * w for x, w in zip(token_basis(turns), (p.C0, p.g)))
    tokens_out = turns * p.out_per_turn
    return {"turns": round(turns, 1), "tokens": int(tokens_in + tokens_out),
            "cost_usd": round((p.k_in * tokens_in + p.k_out * tokens_out) / 1e6, 3),
            "minutes": round(turns * p.sec_per_turn / 60, 1)}


def predict(rating, p=None):
    """Size label plus typical (p50) and worst likely (p90) turns, tokens, cost and time."""
    p = p or Params.load()
    mu = sum(x * w for x, w in zip(turn_features(rating), (p.a, p.b, p.c)))
    return {"size": size(rating.minutes),
            **{q: at_turns(math.exp(mu + z * p.sigma), p) for q, z in QUANTILES.items()}}
