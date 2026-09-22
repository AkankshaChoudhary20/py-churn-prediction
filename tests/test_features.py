"""
Tests for the feature pipeline (features.py): checks that the
ColumnTransformer produces an output of the expected shape and that
categorical columns are one-hot encoded correctly on a small hand-built
sample.
"""

import numpy as np
import pandas as pd
import pytest

from features import build_preprocessor, prepare_X_y, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_sample_df():
    return pd.DataFrame({
        "tenure_months": [1, 24, 60],
        "monthly_charges": [70.0, 55.0, 90.0],
        "total_charges": [70.0, 1320.0, 5400.0],
        "contract_type": ["month-to-month", "one-year", "two-year"],
        "support_tickets_last_90d": [3, 1, 0],
        "avg_weekly_usage": [2.0, 10.0, 20.0],
        "payment_method": ["electronic_check", "credit_card", "bank_transfer"],
        "has_autopay": [False, True, True],
        "churned": [1, 0, 0],
    })


def test_prepare_X_y_shapes_and_dtype():
    df = make_sample_df()
    X, y = prepare_X_y(df)

    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(X) == 3
    assert len(y) == 3
    # has_autopay must be coerced to int 0/1 for the scaler
    assert X["has_autopay"].dtype.kind in ("i", "u")
    assert set(X["has_autopay"].unique()).issubset({0, 1})


def test_preprocessor_output_shape():
    df = make_sample_df()
    X, y = prepare_X_y(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # numeric features: 6 columns (StandardScaler, 1:1)
    # categorical features: contract_type (3 categories) + payment_method (3
    # categories seen in this sample) one-hot encoded
    expected_numeric_cols = len(NUMERIC_FEATURES)
    n_contract_categories = df["contract_type"].nunique()
    n_payment_categories = df["payment_method"].nunique()
    expected_cols = expected_numeric_cols + n_contract_categories + n_payment_categories

    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] == expected_cols


def test_preprocessor_one_hot_encoding_values():
    df = make_sample_df()
    X, y = prepare_X_y(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # locate the one-hot column for contract_type=two-year and confirm only
    # row index 2 (two-year) has a 1 there.
    col_idx = list(feature_names).index("cat__contract_type_two-year")
    col_values = np.asarray(X_transformed)[:, col_idx]
    assert col_values.tolist() == [0.0, 0.0, 1.0]


def test_preprocessor_handles_unknown_category_at_transform_time():
    df = make_sample_df()
    X, y = prepare_X_y(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # A payment_method value not seen during fit should not raise, thanks to
    # handle_unknown="ignore" — it should just produce all-zero one-hot cols
    # for that feature.
    new_row = X.iloc[[0]].copy()
    new_row["payment_method"] = "totally_new_method"
    # should not raise
    transformed = preprocessor.transform(new_row)
    assert transformed.shape[0] == 1
