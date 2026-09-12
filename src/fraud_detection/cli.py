"""Command-line interface (SPEC Section 11).

argparse wiring only, calling into ``pipeline.py`` -- no business logic.
Runnable via ``python -m fraud_detection.cli``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Any

from fraud_detection.config import Config
from fraud_detection.model import VALID_MODEL_TYPES
from fraud_detection.pipeline import run_evaluate, run_generate_data, run_train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-detection",
        description=(
            "Synthetic fraud-detection data generation, training, and "
            "evaluation (local batch CLI)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-data", help="Generate synthetic transaction data and write it to CSV."
    )
    generate.add_argument("--output", type=str, default=None)
    generate.add_argument("--n-samples", type=int, default=None)
    generate.add_argument("--fraud-rate", type=float, default=None)
    generate.add_argument("--seed", type=int, default=None)

    train = subparsers.add_parser("train", help="Train and evaluate a fraud-detection model.")
    train.add_argument("--data-path", type=str, default=None)
    train.add_argument("--model-type", type=str, choices=list(VALID_MODEL_TYPES), default=None)
    train.add_argument("--model-path", type=str, default=None)
    train.add_argument("--test-size", type=float, default=None)
    train.add_argument("--seed", type=int, default=None)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a saved model on held-out data (never retrains)."
    )
    evaluate.add_argument("--data-path", type=str, default=None)
    evaluate.add_argument("--model-path", type=str, default=None)
    evaluate.add_argument("--target-precision", type=float, default=None)

    return parser


def _apply_overrides(config: Config, **overrides: Any) -> Config:
    present = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **present) if present else config


def _print_metrics(label: str, metrics: dict[str, object]) -> None:
    print(f"[{label}] pr_auc={metrics['pr_auc']!r} roc_auc={metrics['roc_auc']!r}")
    print(
        f"[{label}] target_precision={metrics['target_precision']!r} "
        f"recall_at_precision={metrics['recall_at_precision']!r} "
        f"threshold={metrics['threshold']!r}"
    )
    print(f"[{label}] confusion_matrix={metrics['confusion_matrix']!r}")


def _run_generate_data(args: argparse.Namespace) -> int:
    config = _apply_overrides(
        Config(),
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        random_seed=args.seed,
    )
    output = args.output if args.output is not None else config.data_path
    path = run_generate_data(config, output_path=output)
    print(f"Wrote {config.n_samples} rows to {path}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = _apply_overrides(
        Config(),
        model_type=args.model_type,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    metrics = run_train(config, data_path=args.data_path, model_path=args.model_path)
    _print_metrics("train", metrics)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    config = Config()
    metrics = run_evaluate(
        config,
        data_path=args.data_path,
        model_path=args.model_path,
        target_precision=args.target_precision,
    )
    _print_metrics("evaluate", metrics)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate-data":
            return _run_generate_data(args)
        if args.command == "train":
            return _run_train(args)
        if args.command == "evaluate":
            return _run_evaluate(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(  # pragma: no cover - unreachable: subparsers are required
        f"Unknown command: {args.command}"
    )
    return 2  # pragma: no cover - parser.error() above always exits first


if __name__ == "__main__":
    sys.exit(main())
