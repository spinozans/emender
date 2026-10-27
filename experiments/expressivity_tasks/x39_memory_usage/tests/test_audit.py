import math

import pytest
import torch

from experiments.expressivity_tasks.x39_memory_usage.audit import (
    control_receipt, decay_receipt, mechanism_receipt, scan_example, slope,
)


def test_existing_control_is_transcript_at_all_registered_lengths():
    receipt = control_receipt()
    assert receipt['slope_per_head'] == pytest.approx([1.]*4)
    for row in receipt['rows']:
        assert row['peak_live_per_head'] == [row['context']//2]*4
        assert row['write_enabled_fraction'] == 1.
        assert row['successful_allocations_per_token_head'] == .5
        assert row['pressure_fraction_per_window'] == [0., 0.]
        assert row['mean_live_per_head'] is None


def test_decay_is_derived_from_actual_t2_constants():
    receipt = decay_receipt()
    assert math.isclose(receipt['effective_horizon'],128.,abs_tol=.001)
    assert receipt['control_horizon'] > 1_000_000
    _, final, trace, _ = scan_example(decay=receipt['effective_fp32_decay'], tokens=300)
    assert (trace.freed == 2).any()
    assert final.live.sum(-1).max() <= 64


def test_gate_and_semantic_key_mechanism_counterexamples():
    receipt = mechanism_receipt()
    assert receipt['status'] == 'blocked'
    assert receipt['gate_extremes_same_allocation_trace']
    assert receipt['gate_logit_read_gradient_nonzero']
    assert receipt['repeated_vs_random_keys_same_routing']
    assert receipt['zero_residual_allocation_rate'] == 1.


def test_keys_change_reads_without_changing_write_routing():
    random_keys, _, normal_trace, _ = scan_example()
    repeated_keys, _, repeated_trace, _ = scan_example(repeated_keys=True)
    assert torch.equal(normal_trace.alloc_target, repeated_trace.alloc_target)
    assert not torch.equal(random_keys, repeated_keys)


def test_slope_does_not_hide_zero_or_invalid_occupancy():
    assert slope([512,1024,2048,4096], [64]*4) == pytest.approx(0.)
    with pytest.raises(ValueError):
        slope([512,1024], [0,1])
    with pytest.raises(ValueError):
        slope([512,512], [1,2])
