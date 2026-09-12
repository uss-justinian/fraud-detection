"""Central configuration for the pipeline."""

from dataclasses import dataclass, field


@dataclass
class Config:
    random_seed: int = 42
    n_samples: int = 20_000
    fraud_rate: float = 0.015
    test_size: float = 0.25
    model_type: str = "random_forest"  # "random_forest" or "logistic_regression"
    data_path: str = "data/transactions.csv"
    model_path: str = "models/fraud_model.joblib"

    numeric_features: list = field(
        default_factory=lambda: [
            "amount",
            "amount_log",
            "account_age_days",
            "distance_from_home_km",
            "hour_of_day",
            "transactions_last_24h",
        ]
    )
    categorical_features: list = field(
        default_factory=lambda: [
            "merchant_category",
            "is_weekend",
            "is_night",
            "is_foreign",
        ]
    )
    target: str = "is_fraud"
