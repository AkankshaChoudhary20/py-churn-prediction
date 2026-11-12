import json
from predict import build_arg_parser, load_record

def test_customer_json_accepts_utf8_bom(tmp_path):
    record = dict(tenure_months=3, monthly_charges=85.5, total_charges=256.5,
                  contract_type='month-to-month', support_tickets_last_90d=4,
                  avg_weekly_usage=3.2, payment_method='electronic_check', has_autopay=False)
    path = tmp_path / 'customer.json'
    path.write_text(json.dumps(record), encoding='utf-8-sig')
    assert load_record(build_arg_parser().parse_args(['--json', str(path)])) == record
