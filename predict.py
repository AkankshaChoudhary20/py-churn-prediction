"""
predict.py
==========
CLI that loads the saved best model (from models/best_model.txt, selected in
train.py by mean 5-fold CV ROC-AUC) and scores ONE new customer record,
printing the predicted churn probability and predicted label.

Usage (CLI flags):
    python predict.py \\
        --tenure_months 3 \\
        --monthly_charges 85.5 \\
        --total_charges 256.5 \\
        --contract_type month-to-month \\
        --support_tickets_last_90d 4 \\
        --avg_weekly_usage 3.2 \\
        --payment_method electronic_check \\
        --has_autopay false

Usage (JSON file):
    python predict.py --json customer.json

Where customer.json looks like:
    {
      "tenure_months": 3,
      "monthly_charges": 85.5,
      "total_charges": 256.5,
      "contract_type": "month-to-month",
      "support_tickets_last_90d": 4,
      "avg_weekly_usage": 3.2,
      "payment_method": "electronic_check",
      "has_autopay": false
    }
"""

import argparse
import json
import math
import os
import sys

import joblib
import pandas as pd

from features import ALL_FEATURES

MODELS_DIR = "models"


def str2bool(v):
    if isinstance(v, bool):
        return v
    normalized = str(v).strip().lower()
    if normalized in ("true", "1", "yes", "y", "t"):
        return True
    if normalized in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean: true/false, yes/no, or 1/0.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Score a single customer record for churn probability."
    )
    parser.add_argument("--json", type=str, default=None,
                         help="Path to a JSON file with one customer record.")
    parser.add_argument("--model", type=str, default=None,
                         choices=["logistic_regression", "random_forest"],
                         help="Which saved model to use. Defaults to the best "
                              "model recorded in models/best_model.txt.")

    parser.add_argument("--tenure_months", type=int)
    parser.add_argument("--monthly_charges", type=float)
    parser.add_argument("--total_charges", type=float)
    parser.add_argument("--contract_type", type=str,
                         choices=["month-to-month", "one-year", "two-year"])
    parser.add_argument("--support_tickets_last_90d", type=int)
    parser.add_argument("--avg_weekly_usage", type=float)
    parser.add_argument("--payment_method", type=str,
                         choices=["credit_card", "bank_transfer",
                                  "electronic_check", "mailed_check"])
    parser.add_argument("--has_autopay", type=str2bool)

    return parser


def load_record(args) -> dict:
    if args.json:
        try:
            with open(args.json) as f:
                record = json.load(f)
        except (OSError, ValueError) as error:
            print(f"Error: could not load customer JSON: {error}", file=sys.stderr)
            sys.exit(1)
    else:
        record = {
            "tenure_months": args.tenure_months,
            "monthly_charges": args.monthly_charges,
            "total_charges": args.total_charges,
            "contract_type": args.contract_type,
            "support_tickets_last_90d": args.support_tickets_last_90d,
            "avg_weekly_usage": args.avg_weekly_usage,
            "payment_method": args.payment_method,
            "has_autopay": args.has_autopay,
        }

    if not isinstance(record, dict):
        print("Error: customer JSON must be an object containing one record.", file=sys.stderr)
        sys.exit(1)

    missing = [k for k in ALL_FEATURES if record.get(k, None) is None]
    if missing:
        print(f"Error: missing required fields: {missing}", file=sys.stderr)
        print("Provide them via CLI flags or a --json file.", file=sys.stderr)
        sys.exit(1)

    integer_fields = {"tenure_months", "support_tickets_last_90d"}
    numeric_fields = integer_fields | {"monthly_charges", "total_charges", "avg_weekly_usage"}
    for field in sorted(numeric_fields):
        value = record[field]
        valid = type(value) in (int, float) and value >= 0
        if field in integer_fields:
            valid = valid and type(value) is int
        # Avoid an OverflowError when JSON contains an arbitrarily large int.
        try:
            valid = valid and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            kind = "integer" if field in integer_fields else "number"
            print(f"Error: {field} must be a finite nonnegative {kind}.", file=sys.stderr)
            sys.exit(1)

    categories = {
        "contract_type": ("month-to-month", "one-year", "two-year"),
        "payment_method": ("credit_card", "bank_transfer", "electronic_check", "mailed_check"),
    }
    for field, choices in categories.items():
        if not isinstance(record[field], str) or record[field] not in choices:
            print(f"Error: {field} must be one of {', '.join(choices)}.", file=sys.stderr)
            sys.exit(1)

    try:
        record["has_autopay"] = str2bool(record["has_autopay"])
    except argparse.ArgumentTypeError as error:
        print(f"Error: has_autopay: {error}", file=sys.stderr)
        sys.exit(1)
    return record


def resolve_model_name(args) -> str:
    if args.model:
        return args.model
    best_path = os.path.join(MODELS_DIR, "best_model.txt")
    if os.path.exists(best_path):
        with open(best_path) as f:
            return f.read().strip()
    return "random_forest"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    record = load_record(args)
    model_name = resolve_model_name(args)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}. Run train.py first.",
              file=sys.stderr)
        sys.exit(1)

    pipeline = joblib.load(model_path)

    X = pd.DataFrame([record])[ALL_FEATURES].copy()
    X["has_autopay"] = X["has_autopay"].astype(bool).astype(int)

    proba = pipeline.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)

    print(f"Model used: {model_name}")
    print(f"Churn probability: {proba:.4f}")
    print(f"Predicted label: {'CHURN' if label == 1 else 'NO CHURN'} ({label})")


if __name__ == "__main__":
    main()
