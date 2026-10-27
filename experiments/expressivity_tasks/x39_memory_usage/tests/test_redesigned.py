import pytest
import torch

from experiments.expressivity_tasks.x39_memory_usage.redesigned import (
    horizon_receipt, oracle_trace, write_gate_receipt,
)


def test_write_enable_controller_crosses_fixed_threshold_after_cost_optimization():
    torch.set_num_threads(1)
    rows = write_gate_receipt()['rows']
    assert [r['allocation_rate'] for r in rows] == pytest.approx([.5, .5, .3359375, 0.])
    assert all(r['initial_cost_gradient'] > 0 for r in rows[1:])
    assert rows[-1]['routing']['NO-WRITE'] == 128*4


def test_decay_equilibrates_without_budget_saturation():
    torch.set_num_threads(1)
    receipt = horizon_receipt()
    assert receipt['horizon']['effective_horizon'] == pytest.approx(128., abs=.001)
    assert receipt['trace']['routing']['R-CAP'] == 0
    assert receipt['trace']['decay_deaths'] > 0
    assert max(max(v) for v in receipt['trace']['occupancy'][256:]) <= 65


def test_threshold_remains_strict_and_independent_of_append_amplitude():
    torch.set_num_threads(1)
    off = oracle_trace(torch.full((1, 16, 4), .5))
    on = oracle_trace(torch.full((1, 16, 4), .5001))
    assert off['allocation_rate'] == 0.
    assert on['allocation_rate'] > 0.
