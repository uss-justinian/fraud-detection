"""python -m estimator "task text" [--repo PATH]"""

import argparse
import json
from dataclasses import asdict

from estimator.model import predict
from estimator.rating import rate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--repo", help="repository the task is about; its file list gives the rater scope")
    args = ap.parse_args()
    rating = rate(args.task, args.repo)
    print(json.dumps({"rating": asdict(rating), **predict(rating)}, indent=2))


if __name__ == "__main__":
    main()
