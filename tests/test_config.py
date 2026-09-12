"""Tests for fraud_detection.config."""

from fraud_detection.config import Config


def test_config_instantiates_with_no_arguments() -> None:
    config = Config()
    assert config is not None


def test_config_new_fields_have_spec_defaults() -> None:
    config = Config()
    assert config.target_precision == 0.90
    assert config.n_days == 90
    assert config.model_card_path == "models/model_card.md"


def test_config_feature_lists_are_lists_of_str() -> None:
    config = Config()
    assert isinstance(config.numeric_features, list)
    assert isinstance(config.categorical_features, list)
    assert all(isinstance(name, str) for name in config.numeric_features)
    assert all(isinstance(name, str) for name in config.categorical_features)


def test_config_instances_do_not_share_mutable_defaults() -> None:
    a = Config()
    b = Config()
    a.numeric_features.append("mutated")
    assert "mutated" not in b.numeric_features
