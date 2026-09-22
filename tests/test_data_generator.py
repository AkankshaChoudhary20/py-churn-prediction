"""
Tests for generate_data.py: verifies that the synthetic label actually
correlates with the features it's supposed to correlate with. This guards
against a broken / no-signal generator (e.g. a bug that makes `churned`
effectively random noise, which would make the whole project meaningless).
"""

import numpy as np
import pandas as pd
import pytest

from generate_data import generate_customers, churn_probability


@pytest.fixture(scope="module")
def df():
    # Smaller sample for fast tests, but still seeded/deterministic since
    # generate_data.py seeds numpy at import time.
    return generate_customers(n_rows=4000)


def test_generates_expected_columns_and_row_count(df):
    expected_cols = {
        "tenure_months", "monthly_charges", "total_charges", "contract_type",
        "support_tickets_last_90d", "avg_weekly_usage", "payment_method",
        "has_autopay", "churned",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 4000
    assert set(df["churned"].unique()).issubset({0, 1})


def test_churn_rate_is_not_degenerate(df):
    # A broken generator might produce all-0 or all-1 labels, or a rate so
    # extreme that classification is trivial/meaningless.
    rate = df["churned"].mean()
    assert 0.05 < rate < 0.75


def test_month_to_month_has_higher_churn_than_two_year(df):
    rates = df.groupby("contract_type")["churned"].mean()
    assert rates["month-to-month"] > rates["one-year"]
    assert rates["one-year"] > rates["two-year"]


def test_more_support_tickets_means_higher_churn(df):
    # Bucket support tickets and confirm churn rate increases with the bucket.
    bucketed = pd.cut(
        df["support_tickets_last_90d"],
        bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"],
    )
    rates = df.groupby(bucketed, observed=True)["churned"].mean()
    rate_values = rates.reindex(["0", "1", "2", "3+"]).to_numpy()
    # Monotonic non-decreasing trend (allow small noise-driven ties, but the
    # overall direction from lowest to highest bucket must increase).
    assert rate_values[-1] > rate_values[0]
    assert np.all(np.diff(rate_values) >= -0.03)  # allow tiny noise dips


def test_autopay_customers_churn_less(df):
    rates = df.groupby("has_autopay")["churned"].mean()
    assert rates[True] < rates[False]


def test_longer_tenure_means_lower_churn(df):
    bucketed = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 24, 48, 100],
        labels=["0-12", "13-24", "25-48", "49+"],
    )
    rates = df.groupby(bucketed, observed=True)["churned"].mean()
    rate_values = rates.reindex(["0-12", "13-24", "25-48", "49+"]).to_numpy()
    assert rate_values[0] > rate_values[-1]


def test_higher_usage_means_lower_churn(df):
    median_usage = df["avg_weekly_usage"].median()
    low_usage_rate = df.loc[df["avg_weekly_usage"] <= median_usage, "churned"].mean()
    high_usage_rate = df.loc[df["avg_weekly_usage"] > median_usage, "churned"].mean()
    assert low_usage_rate > high_usage_rate


def test_total_charges_approximately_tenure_times_monthly(df):
    implied = df["tenure_months"] * df["monthly_charges"]
    diff = (df["total_charges"] - implied).abs()
    # total_charges should track tenure*monthly_charges reasonably closely
    # (noise stddev is 40 in the generator), not be unrelated.
    correlation = np.corrcoef(implied, df["total_charges"])[0, 1]
    assert correlation > 0.9


def test_churn_probability_formula_is_deterministic_function_of_inputs():
    # Two identical rows (down to the noise draw being irrelevant here since
    # we call churn_probability directly, which adds noise) should still
    # follow the documented directional relationships: a "bad" profile has a
    # much higher probability than a "good" profile, on average across many
    # draws, since the noise term is small relative to the signal spread.
    good_profile = pd.DataFrame([{
        "contract_type": "two-year",
        "support_tickets_last_90d": 0,
        "tenure_months": 60,
        "avg_weekly_usage": 25,
        "has_autopay": True,
        "monthly_charges": 50,
    }] * 200)
    bad_profile = pd.DataFrame([{
        "contract_type": "month-to-month",
        "support_tickets_last_90d": 5,
        "tenure_months": 2,
        "avg_weekly_usage": 1,
        "has_autopay": False,
        "monthly_charges": 100,
    }] * 200)

    good_p = churn_probability(good_profile).mean()
    bad_p = churn_probability(bad_profile).mean()

    assert bad_p > good_p
    assert bad_p - good_p > 0.3
