"""
generate_data.py
=================
Generates a SYNTHETIC customer-churn dataset with realistic, correlated
features and a ground-truth churn probability that is an explicit logistic
function of several of those features (plus noise).

This is NOT real customer data. It is fabricated so that:
  1. It can be published in a public repo with zero privacy/licensing risk.
  2. The relationship between features and the churn label is fully known
     and inspectable (see `churn_probability()` below), which makes it
     possible to sanity-check that the trained models are actually learning
     a real, documented signal rather than memorizing noise.

Run:
    python generate_data.py

Output:
    data/customers.csv  (~5000 rows)

Reproducibility:
    np.random.seed(SEED) is set once at the top of the script, and a single
    numpy Generator seeded from it drives every random draw, so the output
    file is identical on every run.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N_ROWS = 5000

np.random.seed(SEED)
rng = np.random.default_rng(SEED)


def generate_customers(n_rows: int = N_ROWS) -> pd.DataFrame:
    # --- tenure_months: how long the customer has been with the company ---
    # Skewed toward shorter tenures (typical for subscription businesses),
    # capped at 72 months (6 years).
    tenure_months = rng.gamma(shape=2.0, scale=10.0, size=n_rows)
    tenure_months = np.clip(tenure_months, 1, 72).round().astype(int)

    # --- contract_type: influences both charges and churn ---
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_rows,
        p=[0.55, 0.27, 0.18],
    )

    # --- monthly_charges: base price plus a premium for longer contracts ---
    # (customers on longer contracts tend to have bundled/higher-tier plans)
    base_charge = rng.normal(60, 18, size=n_rows)
    contract_premium = np.select(
        [contract_type == "month-to-month", contract_type == "one-year", contract_type == "two-year"],
        [0.0, 6.0, 10.0],
    )
    monthly_charges = np.clip(base_charge + contract_premium, 15, 150).round(2)

    # --- total_charges: approximately tenure * monthly_charges + noise ---
    noise = rng.normal(0, 40, size=n_rows)
    total_charges = np.clip(tenure_months * monthly_charges + noise, 0, None).round(2)

    # --- support_tickets_last_90d: count of support contacts ---
    # Poisson-distributed; customers with billing/usage problems open more.
    support_tickets_last_90d = rng.poisson(lam=1.1, size=n_rows)
    support_tickets_last_90d = np.clip(support_tickets_last_90d, 0, 15)

    # --- avg_weekly_usage: hours/week of product usage ---
    # Negatively related to support tickets (frustrated / disengaged users
    # use the product less) with random variation.
    avg_weekly_usage = np.clip(
        rng.normal(12, 5, size=n_rows) - 0.8 * support_tickets_last_90d,
        0, 60,
    ).round(1)

    # --- payment_method ---
    payment_method = rng.choice(
        ["credit_card", "bank_transfer", "electronic_check", "mailed_check"],
        size=n_rows,
        p=[0.4, 0.25, 0.25, 0.10],
    )

    # --- has_autopay: more likely with credit_card / bank_transfer ---
    autopay_base_prob = np.select(
        [
            payment_method == "credit_card",
            payment_method == "bank_transfer",
            payment_method == "electronic_check",
            payment_method == "mailed_check",
        ],
        [0.85, 0.75, 0.35, 0.15],
    )
    has_autopay = rng.random(n_rows) < autopay_base_prob

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "support_tickets_last_90d": support_tickets_last_90d,
        "avg_weekly_usage": avg_weekly_usage,
        "payment_method": payment_method,
        "has_autopay": has_autopay,
    })

    df["churned"] = simulate_churn_label(df)
    return df


def churn_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Explicit, documented ground-truth churn probability formula.

    This is a logistic function of standardized/derived features:

        z = -1.3
            + 1.55  * is_month_to_month
            - 0.75  * is_two_year
            + 0.70  * (support_tickets_last_90d)          [more tickets -> more churn]
            - 0.050 * tenure_months                        [longer tenure -> less churn]
            - 0.060 * avg_weekly_usage                      [more usage -> less churn]
            - 0.85  * has_autopay                           [autopay -> less churn]
            + 0.012 * monthly_charges                       [pricier plans -> slightly more churn]

        p_churn = sigmoid(z)

    Interpretation (all directions are intentional and documented so the
    "ground truth" relationship is inspectable, not arbitrary):
      * month-to-month contracts strongly increase churn risk
      * two-year contracts strongly decrease churn risk
      * more support tickets in the last 90 days increases churn risk
      * longer tenure decreases churn risk
      * higher weekly product usage (engagement) decreases churn risk
      * autopay decreases churn risk (friction to cancel / stickier billing)
      * higher monthly charges slightly increase churn risk (price sensitivity)

    Gaussian noise is added to `z` before the sigmoid so the label is not a
    deterministic function of the features (real churn always has an
    irreducible random component), which keeps the classification problem
    realistic (ROC-AUC well below 1.0 but well above 0.5).
    """
    is_month_to_month = (df["contract_type"] == "month-to-month").astype(float)
    is_two_year = (df["contract_type"] == "two-year").astype(float)
    autopay = df["has_autopay"].astype(float)

    z = (
        -1.3
        + 1.55 * is_month_to_month
        - 0.75 * is_two_year
        + 0.70 * df["support_tickets_last_90d"]
        - 0.050 * df["tenure_months"]
        - 0.060 * df["avg_weekly_usage"]
        - 0.85 * autopay
        + 0.012 * df["monthly_charges"]
    )

    # irreducible random noise so churn isn't a deterministic function of X
    z = z + rng.normal(0, 0.55, size=len(df))

    p = 1.0 / (1.0 + np.exp(-z))
    return p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)


def simulate_churn_label(df: pd.DataFrame) -> np.ndarray:
    p = churn_probability(df)
    draws = rng.random(len(df))
    return (draws < p).astype(int)


def main():
    df = generate_customers(N_ROWS)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "customers.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Overall churn rate: {df['churned'].mean():.3f}")
    print("\nChurn rate by contract_type:")
    print(df.groupby("contract_type")["churned"].mean())
    print("\nChurn rate by has_autopay:")
    print(df.groupby("has_autopay")["churned"].mean())


if __name__ == "__main__":
    main()
