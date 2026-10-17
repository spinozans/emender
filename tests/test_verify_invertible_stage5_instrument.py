"""X5D scalar decoding and exact depth/prefix/loss contracts."""
import numpy as np
import pytest

from experiments.expressivity_tasks.verify_invertible_stage5_instrument import decode_batch
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask
from experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m import padded_forward


@pytest.mark.parametrize('depth', [1, 2, 4, 8, 16, 32, 64, 128])
def test_scalar_decode_checks_driver_count_joint_targets_and_init_convention(depth):
    batch = InvertibleQuadraticTask(61).generate_batch(32, depth+2, np.random.default_rng(0))
    receipt = decode_batch(batch, depth)
    assert receipt['shape'] == [32, depth+2]
    assert receipt['supervised_positions'] == 32*depth
    assert len(receipt['decoded'][0]['drivers']) == depth
    with pytest.raises(AssertionError):
        decode_batch(batch, depth+1)


def test_hand_computed_four_driver_example():
    x = np.array([[112, 160, 53, 8, 35, 44]], dtype=np.int64)
    y = np.array([[0, 0, 2341, 1429, 1588, 135]], dtype=np.int64)
    mask = np.array([[False, False, True, True, True, True]])
    row = decode_batch((x, y, mask), 4)['decoded'][0]
    assert row['initial'] == [51, 38]
    assert [s['state'] for s in row['composed']] == [[38, 23], [23, 26], [26, 2], [2, 13]]


@pytest.mark.parametrize('corruption', ['target', 'prefix_mask', 'driver_mask', 'init', 'driver'])
def test_scalar_decode_fails_closed_on_invalid_targets_or_visibility(corruption):
    x, y, mask = InvertibleQuadraticTask(61).generate_batch(2, 6, np.random.default_rng(0))
    if corruption == 'target':
        y[0, 2] = (y[0, 2]+1) % 3721
    elif corruption == 'prefix_mask':
        mask[0, 0] = True
    elif corruption == 'driver_mask':
        mask[0, 2] = False
    elif corruption == 'init':
        x[0, 0] = 0
    else:
        x[0, 2] = 61
    with pytest.raises(AssertionError):
        decode_batch((x, y, mask), 4)


def test_training_stream_replay_distinguishes_four_from_128_drivers():
    from experiments.expressivity_tasks.audit_invertible_stage5_instrument import replay
    rng = np.random.default_rng(0)
    task = InvertibleQuadraticTask(61)
    for _ in range(3):
        task.generate_batch(32, 6, rng)
    state, digest = replay(0, 4, 3)
    assert state == rng.bit_generator.state
    assert replay(0, 4, 3) == (state, digest)
    other_state, other_digest = replay(0, 128, 3)
    assert other_state != state and other_digest != digest


def test_padded_forward_preserves_prefix_visibility_and_removes_all_pad_logits():
    import torch
    observed = []
    def causal_model(tokens):
        observed.append(tokens.clone())
        return tokens.cumsum(1).unsqueeze(-1).float()
    x = torch.tensor([[61, 122, 3, 4, 5, 6]])
    full = padded_forward(causal_model, x)
    first = padded_forward(causal_model, x[:, :3])
    assert full.shape == (1, 6, 1) and first.shape == (1, 3, 1)
    assert all(t.shape == (1, 16) for t in observed)
    torch.testing.assert_close(full[:, :3], first)
    assert full[0, 2, 0].item() == 61+122+3
    assert (observed[0][:, 6:] == 0).all()
