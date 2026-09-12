"""Tests for fraud_detection.cli (SPEC Section 11 / QA_PLAN CLI acceptance
criteria)."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from fraud_detection import cli


def test_help_exits_zero(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


@pytest.mark.parametrize("subcommand", ["generate-data", "train", "evaluate"])
def test_subcommand_help_exits_zero(subcommand: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([subcommand, "--help"])
    assert exc_info.value.code == 0


def test_generate_data_writes_csv(tmp_path: Path) -> None:
    output = tmp_path / "t.csv"
    exit_code = cli.main(
        ["generate-data", "--n-samples", "500", "--seed", "1", "--output", str(output)]
    )
    assert exit_code == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 500


def test_train_then_evaluate_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "t.csv"
    model_path = tmp_path / "m.joblib"

    gen_code = cli.main(
        ["generate-data", "--n-samples", "500", "--seed", "1", "--output", str(data_path)]
    )
    assert gen_code == 0

    train_code = cli.main(
        [
            "train",
            "--data-path",
            str(data_path),
            "--model-path",
            str(model_path),
            "--model-type",
            "logistic_regression",
        ]
    )
    assert train_code == 0
    assert model_path.exists()
    assert joblib.load(model_path) is not None
    assert (tmp_path / "models" / "model_card.md").exists()

    evaluate_code = cli.main(
        ["evaluate", "--data-path", str(data_path), "--model-path", str(model_path)]
    )
    assert evaluate_code == 0


def test_train_both_model_types_succeed_no_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "t.csv"
    cli.main(["generate-data", "--n-samples", "500", "--seed", "1", "--output", str(data_path)])

    for model_type in ("logistic_regression", "random_forest"):
        model_path = tmp_path / f"m_{model_type}.joblib"
        exit_code = cli.main(
            [
                "train",
                "--data-path",
                str(data_path),
                "--model-path",
                str(model_path),
                "--model-type",
                model_type,
            ]
        )
        assert exit_code == 0
        assert model_path.exists()


def test_evaluate_prints_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "t.csv"
    model_path = tmp_path / "m.joblib"
    cli.main(["generate-data", "--n-samples", "500", "--seed", "1", "--output", str(data_path)])
    cli.main(
        [
            "train",
            "--data-path",
            str(data_path),
            "--model-path",
            str(model_path),
            "--model-type",
            "logistic_regression",
        ]
    )
    capsys.readouterr()  # discard generate/train output

    exit_code = cli.main(
        ["evaluate", "--data-path", str(data_path), "--model-path", str(model_path)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pr_auc" in out
    assert "roc_auc" in out
    assert "confusion_matrix" in out


def test_train_invalid_model_type_exits_nonzero_naming_choices(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["train", "--model-type", "not-a-model"])
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "logistic_regression" in err
    assert "random_forest" in err


def test_train_missing_data_file_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = cli.main(
        [
            "train",
            "--data-path",
            str(tmp_path / "missing.csv"),
            "--model-path",
            str(tmp_path / "m.joblib"),
        ]
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "Error" in err


def test_evaluate_missing_model_file_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    data_path = tmp_path / "t.csv"
    cli.main(["generate-data", "--n-samples", "200", "--output", str(data_path)])
    capsys.readouterr()

    exit_code = cli.main(
        [
            "evaluate",
            "--data-path",
            str(data_path),
            "--model-path",
            str(tmp_path / "missing.joblib"),
        ]
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "Error" in err


def test_no_predict_subcommand_exists() -> None:
    """Confirms SPEC's non-goal (no predict/scoring CLI command) is honored."""
    with pytest.raises(SystemExit):
        cli.main(["predict", "--data-path", "x.csv"])
