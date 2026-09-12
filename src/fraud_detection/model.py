"""Model pipeline construction (SPEC Section 8).

No training logic beyond construction — fitting happens by calling
``.fit()`` on the returned, untrained ``sklearn.pipeline.Pipeline``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.config import Config

VALID_MODEL_TYPES = ("logistic_regression", "random_forest")


def build_pipeline(config: Config) -> Pipeline:
    """Build an untrained sklearn Pipeline for ``config.model_type``.

    Swapping models is a config change only: no branching on model_type
    happens anywhere outside this function.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.categorical_features,
            ),
        ]
    )

    if config.model_type == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.random_seed,
        )
    elif config.model_type == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=config.random_seed,
        )
    else:
        raise ValueError(
            "config.model_type must be one of "
            f"{VALID_MODEL_TYPES!r}, got {config.model_type!r}"
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
