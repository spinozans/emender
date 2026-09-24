import pytest
import torch
from ndm.bf16_rounding_probe import stochastic_round_bf16


def generator(seed=2718):
    return torch.Generator().manual_seed(seed)


def test_exact_bf16_values_and_signs_are_preserved():
    x = torch.tensor([0.0, -0.0, 0.01, -0.01, 1, -1, 1e-30, -1e-30], dtype=torch.bfloat16)
    actual = stochastic_round_bf16(x.float(), generator=generator())
    assert torch.equal(actual.view(torch.int16), x.view(torch.int16))


def test_small_update_rounds_away_in_rne_but_mean_survives_stochastically():
    for sign in (-1, 1):
        base = torch.tensor(sign*0.01, dtype=torch.bfloat16)
        value = base.float() - sign*2e-6
        assert value.bfloat16() == base
        samples = stochastic_round_bf16(value.expand(262144), generator=generator())
        assert 0 < int((samples != base).sum()) < samples.numel()
        assert abs(float(samples.double().mean()) - float(value)) < 2e-7


def test_rng_resume_reproduces_and_does_not_consume_global_rng():
    global_state = torch.get_rng_state().clone()
    rng = generator()
    x = torch.full((8192,), 0.010001, dtype=torch.float32)
    state = rng.get_state().clone()
    first = stochastic_round_bf16(x, generator=rng)
    restored = generator(0); restored.set_state(state)
    assert torch.equal(first, stochastic_round_bf16(x, generator=restored))
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 3.4e38])
def test_nonfinite_or_overflow_rejected(value):
    with pytest.raises(ValueError):
        stochastic_round_bf16(torch.tensor([value]), generator=generator())


def test_non_fp32_rejected():
    with pytest.raises(ValueError):
        stochastic_round_bf16(torch.ones(2,dtype=torch.bfloat16), generator=generator())
