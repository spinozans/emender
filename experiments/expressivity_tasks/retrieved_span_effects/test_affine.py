"""No data, fitted weights or accelerator. Alternative-hit mixing is not time composition."""

import pytest
import torch

from .affine import compile_gdn2, merged_read


def replay(controls, seed):
    # Independent serial reference: never constructs an A matrix.
    state = seed
    for key, decay, erase, write in zip(*controls, strict=True):
        decayed = decay[:, None] * state
        error = write - (erase * key) @ decayed
        state = decayed + key[:, None] * error[None, :]
    return state


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-12), (torch.float32, 3e-6)])
def test_compose_then_pool_matches_parallel_replay_and_derivatives(dtype, tol):
    gen = torch.Generator().manual_seed(921)

    def draw(*shape):
        return torch.randn(shape, generator=gen, dtype=dtype, requires_grad=True)

    raw = [draw(7, width) for width in (3, 3, 3, 5, 5)]
    table = (
        torch.nn.functional.normalize(raw[0]),
        raw[1].sigmoid(),
        raw[2].sigmoid(),
        raw[3].sigmoid() * raw[4],
    )
    spans = [
        [1, 4, 2],
        [4, 2],
        [6, 6, 3, 0],
        [1, 4, 2],
    ]  # same hit twice retains both masses
    controls = [tuple(x[ids] for x in table) for ids in spans]
    effects = [compile_gdn2(*c) for c in controls]

    def close(x, y):
        torch.testing.assert_close(x, y, atol=tol, rtol=tol)

    for c, h in zip(controls, effects, strict=True):
        close(h.b, replay(c, torch.zeros_like(h.b)))
        left, right = (
            compile_gdn2(*(v[sl] for v in c)) for sl in (slice(1), slice(1, None))
        )
        close(right.after(left).a, h.a)
        close(right.after(left).b, h.b)
    weights = draw(len(spans))
    seed = draw(3, 5)
    q = draw(3)
    probe = draw(5).detach()
    got = merged_read(effects, weights.exp(), seed, q)
    ref = sum(
        p * (q @ replay(c, seed))
        for p, c in zip(weights.softmax(0), controls, strict=True)
    )
    close(got, ref)
    inputs = (*raw, weights, seed, q)
    grad = torch.autograd.grad(got @ probe, inputs, retain_graph=True)
    for a, b in zip(grad, torch.autograd.grad(ref @ probe, inputs), strict=True):
        close(a, b)
    # Reuse the SAME summaries on a different seed, no re-encoding of passage tokens.
    seed2 = seed.detach().roll(1, 0) - 0.5
    for c, h in zip(controls, effects, strict=True):
        close(h.apply(seed2), replay(c, seed2))


def test_blank_state_can_destroy_the_information_that_distinguishes_two_passages():
    eye = torch.eye(2, dtype=torch.double)
    one = torch.ones((1, 2), dtype=eye.dtype)
    # Two tokens erase different rows and write zero. Blank terminal states coincide.
    a, b = [compile_gdn2(k[None], one, one, 0 * one) for k in eye]
    assert torch.equal(a.b, b.b) and not torch.equal(a.a, b.a)
    torch.testing.assert_close(eye[0] @ a.apply(eye), torch.zeros_like(eye[0]))
    torch.testing.assert_close(eye[0] @ b.apply(eye), eye[0])
    # This is observability of the seeded effect, not a language prediction result.


def test_no_hit_vs_an_empty_span_and_invalid_masses():
    e = torch.empty(0, 2)
    h = compile_gdn2(e, e, e, e)
    s = torch.eye(2)
    q = torch.ones(2)
    torch.testing.assert_close(merged_read([h], torch.ones(1), s, q), q)
    big = torch.full((2,), torch.finfo(s.dtype).max, dtype=s.dtype)
    torch.testing.assert_close(merged_read([h, h], big, s, q), q)
    for hits, p in [([], torch.empty(0)), ([h], torch.zeros(1))]:
        assert torch.count_nonzero(merged_read(hits, p, s, q)) == 0
    for p in [torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.ones(2)]:
        with pytest.raises(ValueError):
            merged_read([h], p, s, q)


def test_nonlinear_state_does_not_inherit_the_affine_shortcut():
    # One scalar saturating edit. A map agreeing at 0 and 1 need not agree at 2.
    def f(x):
        return torch.tanh(torch.tensor(x, dtype=torch.float64) + 0.4)

    assert abs(f(2) - (f(0) + 2 * (f(1) - f(0)))) > 0.1
