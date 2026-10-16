import argparse
import json

import pytest

from predict import build_arg_parser, load_record, str2bool


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("false", False), ("NO", False), ("0", False), ("yes", True)])
def test_boolean_conversion(value, expected):
    assert str2bool(value) is expected


@pytest.mark.parametrize("value", ["maybe", "", "2", None])
def test_invalid_boolean_is_not_silently_false(value):
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool(value)


def test_invalid_cli_boolean_exits():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--has_autopay", "maybe"])


def test_json_false_string_is_normalized_before_model_conversion(tmp_path):
    record = {"tenure_months": 3, "monthly_charges": 85.5, "total_charges": 256.5,
              "contract_type": "month-to-month", "support_tickets_last_90d": 4,
              "avg_weekly_usage": 3.2, "payment_method": "electronic_check", "has_autopay": "false"}
    path = tmp_path / "customer.json"
    path.write_text(json.dumps(record))
    args = build_arg_parser().parse_args(["--json", str(path)])
    assert load_record(args)["has_autopay"] is False
    record["has_autopay"] = "maybe"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        load_record(args)
