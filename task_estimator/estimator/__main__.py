"""Estimate a task: python -m estimator "task text" [--repo PATH]"""

import argparse
import json

from estimator.model import estimate
from estimator.rubric import rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--repo", help="repository the task is about, for scope context")
    a = ap.parse_args()
    scores = rate(a.task, a.repo)
    print(json.dumps({"rubric": scores, "estimate": estimate(scores)}, indent=2))


if __name__ == "__main__":
    main()
