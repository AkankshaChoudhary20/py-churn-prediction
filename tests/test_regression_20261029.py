import pytest
from predict import build_arg_parser, load_record

def test_duplicate_customer_fields_are_rejected_with_field_name(tmp_path, capsys):
    path = tmp_path / 'customer.json'
    path.write_text('{"tenure_months": 3, "tenure_months": 9, "monthly_charges": 85.5, "total_charges": 256.5, "contract_type": "month-to-month", "support_tickets_last_90d": 4, "avg_weekly_usage": 3.2, "payment_method": "electronic_check", "has_autopay": false}')
    args = build_arg_parser().parse_args(['--json', str(path)])
    with pytest.raises(SystemExit) as error:
        load_record(args)
    assert error.value.code == 1
    assert 'Duplicate customer JSON field: tenure_months' in capsys.readouterr().err
