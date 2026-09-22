"""
features.py
============
Defines the scikit-learn ColumnTransformer used to preprocess raw customer
records into a numeric matrix suitable for model training:
  * numeric columns -> StandardScaler
  * categorical columns -> OneHotEncoder
  * boolean column (has_autopay) -> passed through as numeric (0/1) after
    scaling like the other numerics.

The same ColumnTransformer is wrapped inside each model's sklearn Pipeline
in train.py, so preprocessing is always fit only on the training fold /
training set, never leaking test data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_last_90d",
    "avg_weekly_usage",
    "has_autopay",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "churned"


def build_preprocessor() -> ColumnTransformer:
    """Builds the ColumnTransformer: OneHotEncoder for categoricals,
    StandardScaler for numerics (including the boolean has_autopay, which
    is 0/1 and scales fine as a numeric feature)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def prepare_X_y(df):
    """Given the raw customers DataFrame, returns (X, y) where X has
    has_autopay coerced to int (0/1) so StandardScaler works on it."""
    X = df[ALL_FEATURES].copy()
    X["has_autopay"] = X["has_autopay"].astype(int)
    y = df[TARGET].astype(int)
    return X, y
