"""Tests for fraud_detection.model (SPEC Section 8 / QA_PLAN model
acceptance criteria)."""

from dataclasses import replace

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from fraud_detection.config import Config
from fraud_detection.data import generate_transactions
from fraud_detection.features import add_features
from fraud_detection.model import build_pipeline


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_build_pipeline_returns_unfitted_pipeline(model_type: str) -> None:
    config = Config(model_type=model_type)
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps["classifier"])


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_class_weight_is_balanced(model_type: str) -> None:
    config = Config(model_type=model_type)
    pipeline = build_pipeline(config)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.class_weight == "balanced"


def test_invalid_model_type_raises_value_error() -> None:
    config = Config(model_type="svm")
    with pytest.raises(ValueError):
        build_pipeline(config)


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_pipeline_fits_and_predicts_proba(model_type: str) -> None:
    config = replace(Config(model_type=model_type), n_samples=500, random_seed=3)
    raw = generate_transactions(config)
    featured = add_features(raw)
    feature_cols = config.numeric_features + config.categorical_features

    pipeline = build_pipeline(config)
    pipeline.fit(featured[feature_cols], featured[config.target])
    proba = pipeline.predict_proba(featured[feature_cols])

    assert proba.shape == (len(featured), 2)
