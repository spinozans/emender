import json
import pytest
from experiments.expressivity_tasks.x43_fork_resolved import c14_protocol_gap_summary as summary


def test_both_siblings_halve_payload_and_oom_is_not_a_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'RECEIPTS', tmp_path)
    traffic = {name: summary.audit(tokens=128, peak=peak)['one_pass_payload_bytes']
               for name, peak in [('unbounded',257),('budgeted',48)]}
    for slots, name in [(257,'unbounded'), (48,'budgeted')]:
        row = dict(status='measured', median_ms=100+traffic[name]*1e-4,
                   peak_occupancy=min(128,slots), slots=slots,
                   max_memory_bytes=2000, max_reserved_bytes=3000, device_total_bytes=4000)
        (tmp_path/f'stress{slots}_t128_fixed.json').write_text(json.dumps(row))
    result = summary.summarize(128)
    bf = result['conditional_bf16_fit']
    assert bf['common_ms'] == pytest.approx(100)
    assert bf['unbounded_over_budgeted'] == pytest.approx(
        (100+traffic['unbounded']*5e-5)/(100+traffic['budgeted']*5e-5))
    row['status'] = 'failed'
    (tmp_path/'stress48_t128_fixed.json').write_text(json.dumps(row))
    assert 'full_step_unbounded_over_budgeted' not in summary.summarize(128)


def test_negative_common_cost_rejects_full_step_precision_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'RECEIPTS', tmp_path)
    for slots, ms in [(257,2000), (48,1000)]:
        row = dict(status='measured', median_ms=ms, peak_occupancy=min(128,slots),
                   slots=slots, max_memory_bytes=2000, max_reserved_bytes=3000,
                   device_total_bytes=4000)
        (tmp_path/f'stress{slots}_t128_fixed.json').write_text(json.dumps(row))
    result = summary.summarize(128)
    assert result['full_step_unbounded_over_budgeted'] == 2
    assert not result['conditional_bf16_fit']['physically_nonnegative']
    assert 'full_step_ms' not in result['conditional_bf16_fit']
