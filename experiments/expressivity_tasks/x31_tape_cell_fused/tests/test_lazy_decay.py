import random
import struct

import pytest
import torch

from experiments.expressivity_tasks.x30_tape_cell.float_decay import advance, death, f32
from experiments.expressivity_tasks.x31_tape_cell_fused.lazy_decay import evaluate_decay
from experiments.expressivity_tasks.x31_tape_cell_fused.device_schedule import exercise_schedule
from .test_parity import device


@pytest.mark.parametrize('decay', [.125, .5, .8, .99, .9995, 1.-2**-24, 1.])
def test_exact_device_interval_jumps(decay):
    dev = device()
    rng = random.Random(731)
    strengths = [1., .125, .50000006, 1.00000012]
    strengths += [f32(rng.uniform(.002, 2.)) for _ in range(124)]
    steps = [0, 1, 2, 127] + [rng.randrange(0, 20000) for _ in range(124)]
    threshold = f32(.001)
    expected = [advance(s, decay, t) for s, t in zip(strengths, steps)]
    schedules = [death(s, decay, threshold) for s in strengths]
    result, horizon, terminal = evaluate_decay(
        torch.tensor(strengths, device=dev), torch.tensor(steps, device=dev), decay, threshold)
    assert result.tolist() == expected
    assert horizon.tolist() == [(-1 if h is None else h) for h, _ in schedules]
    assert terminal.tolist() == [s for _, s in schedules]


def test_boundary_and_popped_reference():
    dev = device()
    decay, threshold = f32(.9995), f32(.001)
    strengths = [f32(1.-p) for p in [.01, .3, .6, .9]]
    # Immediate neighbors of threshold and nonunit pop references exercise
    # exact <= boundaries; the horizon must match, not merely be allclose.
    bits = struct.unpack('<I', struct.pack('<f', threshold))[0]
    strengths += [struct.unpack('<f', struct.pack('<I', bits+i))[0] for i in [1, 2, 16, 1024]]
    horizons = [death(s, decay, threshold)[0] for s in strengths]
    for offset in [-1, 0, 1]:
        steps = [max(0, t+offset) for t in horizons]
        result, h, terminal = evaluate_decay(
            torch.tensor(strengths, device=dev), torch.tensor(steps, device=dev), decay, threshold)
        assert h.tolist() == horizons
        assert result.tolist() == [advance(s, decay, t) for s, t in zip(strengths, steps)]
        assert terminal.tolist() == [death(s, decay, threshold)[1] for s in strengths]


@pytest.mark.parametrize('decay', [.8, .9995, 1.])
def test_cancellable_device_schedule(decay):
    dev = device()
    n, s, e = 2, 37, 180
    rng = random.Random(731)
    ids, strengths, times, boundaries, expected = [], [], [], [], []
    for store in range(n):
        pending = {}
        row_ids, row_strengths, row_times, row_boundaries, row_expected = [], [], [], [], []
        for event in range(e):
            # Repeated IDs replace old schedules; tied times are slot-ordered.
            slot = rng.randrange(s) if event < e-1 else -1
            ref = f32(rng.choice([1., .3, .01, .001001]))
            now = event
            boundary = now if event < e-1 else 100000
            if slot >= 0:
                horizon, _ = death(ref, decay, .001)
                pending.pop(slot, None)
                if horizon is not None:
                    pending[slot] = now+horizon
            dead = sorted((time, slot_id) for slot_id, time in pending.items() if time <= boundary)
            row_expected.append([slot_id for _, slot_id in dead])
            for _, slot_id in dead:
                del pending[slot_id]
            row_ids.append(slot); row_strengths.append(ref)
            row_times.append(now); row_boundaries.append(boundary)
        ids.append(row_ids); strengths.append(row_strengths)
        times.append(row_times); boundaries.append(row_boundaries); expected.append(row_expected)
    output, counts, heap, position, due, sizes = exercise_schedule(
        torch.tensor(ids, device=dev), torch.tensor(strengths, device=dev),
        torch.tensor(times, device=dev), torch.tensor(boundaries, device=dev), s, decay, .001)
    for store in range(n):
        for event in range(e):
            assert output[store,event,:counts[store,event]].tolist() == expected[store][event]
    assert sizes.eq(0).all()
    assert position.eq(-1).all()
