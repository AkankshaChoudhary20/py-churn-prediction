import json

import pytest

from predict import build_arg_parser, load_record


def valid_record():
    return {'tenure_months': 3, 'monthly_charges': 85.5, 'total_charges': 256.5,
            'contract_type': 'month-to-month', 'support_tickets_last_90d': 4,
            'avg_weekly_usage': 3.2, 'payment_method': 'electronic_check', 'has_autopay': False}


@pytest.mark.parametrize('field,value', [
    ('tenure_months', -1), ('tenure_months', 1.5), ('tenure_months', True),
    ('support_tickets_last_90d', -1), ('support_tickets_last_90d', 'four'),
    ('monthly_charges', float('nan')), ('monthly_charges', float('inf')),
    ('total_charges', -1), ('avg_weekly_usage', []), ('avg_weekly_usage', True),
    ('contract_type', 'unknown'), ('payment_method', []),
])
def test_invalid_json_fields_have_clear_errors(field, value, tmp_path, capsys):
    record = valid_record()
    record[field] = value
    path = tmp_path / 'customer.json'
    path.write_text(json.dumps(record))
    args = build_arg_parser().parse_args(['--json', str(path)])
    with pytest.raises(SystemExit) as error:
        load_record(args)
    assert error.value.code == 1
    assert field in capsys.readouterr().err


def test_invalid_cli_numbers_are_also_rejected(capsys):
    flags = [part for key, value in valid_record().items() for part in ('--' + key, str(value))]
    flags[flags.index('--monthly_charges') + 1] = 'nan'
    with pytest.raises(SystemExit):
        load_record(build_arg_parser().parse_args(flags))
    assert 'monthly_charges' in capsys.readouterr().err


def test_valid_json_record_is_preserved(tmp_path):
    record = valid_record()
    path = tmp_path / 'customer.json'
    path.write_text(json.dumps(record))
    assert load_record(build_arg_parser().parse_args(['--json', str(path)])) == record
