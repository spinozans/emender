import numpy as np
import pytest
from experiments.expressivity_tasks.x30_tape_cell.float_decay import advance, death, f32


@pytest.mark.parametrize('decay', [.1, .5, .8, .9, .999995, .99999994, 1.])
def test_exact_jump_against_scalar_native_multiply(decay):
    rng = np.random.default_rng(4)
    for start in [1., .001, *rng.uniform(.01, 1., 8)]:
        strength = np.float32(start)
        d = np.float32(decay)
        for n in range(1, 3001):
            strength = np.float32(strength*d)
            if n in (1, 7, 9, 50, 128, 1024, 3000):
                assert advance(f32(start), f32(decay), n) == float(strength)


@pytest.mark.parametrize('decay, threshold', [(.8, .20971521735191345), (.9, .1), (.999995, .001)])
def test_death_horizon_and_boundary_exact(decay, threshold):
    strength = np.float32(1.)
    d = np.float32(decay)
    threshold = np.float32(threshold)
    n = 0
    while strength > threshold:
        strength = np.float32(strength*d)
        n += 1
    assert death(1., f32(decay), float(threshold)) == (n, float(strength))
    assert advance(1., f32(decay), n-1) > threshold
    assert advance(1., f32(decay), n) <= threshold
