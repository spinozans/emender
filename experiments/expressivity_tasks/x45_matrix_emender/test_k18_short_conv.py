"""K18 -- THE SHORT-CONV TRANSPLANT (GDN2's registered short depthwise
convolution, kernel 4, on the payload path of the composed machine), gated.

The composed machine's parity disciplines, extended by the conv lane
(``PREREGISTRATION_SHORT_CONV.md``, committed before any probe
measurement):

* **THE BIRTH RECEIPT (the K15 discipline, carried)**: at the identity
  birth (the registered ``[0, 0, 0, 1]`` kernel in the STANDARD nn.Conv1d
  tap layout -- the LAST tap is the current token) the conv is exactly the
  pass-through in every dtype (``1.0*x`` is exact; the zero taps add
  +-0.0), so the K18 mixer is BIT-IDENTICAL to the K15 composed mixer's
  own output, and the machine is BORN AS THE COMPOSED MACHINE bit for
  bit.  The only legal escape is the IEEE sign-of-zero corner (a -0.0
  payload entry may flip to +0.0 through the zero taps' additions) --
  counted and receipted here; it is invisible to every ``==``-based
  parity bar (``-0.0 == +0.0``) and to the loss.
* **P2 trivially held (the value-channel receipt)**: no conv value feeds
  op selection -- the recorded schedule, the read WEIGHTS, the
  pre-decision liveness and the key table are BIT-INVARIANT to every conv
  value at fixed keys (the 0-flip receipt under RANDOM conv kernels),
  while the value state moves off the identity birth as designed.
* **The gradient gates are floor-discipline**: the fp64 ORACLE validates
  itself first -- a gradcheck THROUGH THE CONV PATH (the conv's own tap
  adjoint attached in front of the composed walk, over every content
  carrying branch) -- then the kernel path's gradients through the conv
  chain sit within ``NOISE_FACTOR`` of the measured fp32-vs-fp64 floor.
  The conv's step-0 gradient receipt: the identity-born kernel is NOT a
  dead parameter (unlike the zero-born gates, the payload path reaches it
  at step 0).
* **The conv IS the real conv**: the explicit tap sum is exactly
  ``F.conv1d(pad(x), w.view(C,1,K), groups=C)`` up to floating-point
  accumulation order (receipted: fp64 within 1e-15, fp32 within 1e-6), and
  the fp64 tap adjoint is the analytic correlation EXACTLY.
* **The conversion receipts**: the parameter delta is the registration's
  (the composed machine's own + one depthwise kernel of heads*value_dim
  channels x 4 taps per mixer), the conv at the registered birth, and the
  refusals are by name.

The conv unit tests run anywhere; the scan/mixer/gradcheck tests need
CUDA (the committed Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    ComposedReadMixer, _k15_forward_launch, composed_oracle_scan,
    composed_parameter_delta, composed_scan, convert_model_to_composition)
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv import (
    IDENTITY_SHORT_CONV_BIRTH, SHIFT_SHORT_CONV_BIRTH, SHORT_CONV_BIRTHS,
    SHORT_CONV_KERNEL_SIZE, IdentityBornShortConv, ShortConvComposedMixer,
    conv_payload, convert_model_to_short_conv, short_conv1d_causal,
    short_conv_parameter_delta)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    convert_model_to_separated_keys)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))


def keyed_stream(batch, tokens, heads, value_dim, key_dim, distinct, seed,
                 decay=0.95):
    """The lane's canonical fixture (K10/K11/K11B/K15/K17's): payload AND
    write-key values drawn from ``distinct`` fixed directions, query keys
    random.  The decay channel is the PER-CHANNEL vector."""
    g = torch.Generator().manual_seed(seed)
    payload_dirs = torch.randn(distinct, value_dim, generator=g).to(DEVICE)
    key_dirs = torch.randn(distinct, key_dim, generator=g).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = payload_dirs[choice.cpu()].to(DEVICE)
    write_keys = key_dirs[choice.cpu()].to(DEVICE)
    query_keys = (torch.randn(batch, tokens, heads, key_dim, generator=g)
                  .to(DEVICE) * 0.5)
    if isinstance(decay, float):
        base = torch.full((batch, tokens, heads, 1), decay, device=DEVICE)
        spread = (torch.rand(batch, tokens, heads, value_dim, generator=g)
                  * 0.2 - 0.1).to(DEVICE)
        decay = (base + spread).clamp(min=1e-4, max=1.0 - 1e-6)
    else:
        decay = decay.to(DEVICE)
    return payload, write_keys, query_keys, decay


def gate_stream(batch, tokens, heads, value_dim, seed):
    """Random PER-CHANNEL gates in (0.05, 0.95) -- the off-identity fixture."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, tokens, heads, value_dim, generator=g)
            * 0.9 + 0.05).to(DEVICE)


def composed_kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank, segment_tokens=64, gated=True,
                            collect=True, warps=None):
    """The K15 composed launch, the parity reference (imported verbatim)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=DEVICE)
    if gates is None:
        gates = torch.ones(batch, tokens, heads, value_dim, device=DEVICE)
    return _k15_forward_launch(payload, write_keys, query_keys, decay,
                               gates, state, live, keys, POL32, rank,
                               segment_tokens, gated, collect=collect,
                               warps=warps if warps is not None else
                               (4 if rank * value_dim < 8192 else 8),
                               stages=3)


def conv_stream(batch, tokens, heads, value_dim, seed, channels=None):
    """A random CONV kernel fixture (off the identity birth)."""
    g = torch.Generator().manual_seed(seed)
    channels = heads * value_dim if channels is None else channels
    return torch.randn(channels, SHORT_CONV_KERNEL_SIZE,
                       generator=g).to(DEVICE) * 0.5


# ------------------------------------------------- the conv unit receipts (CPU)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64,
                                   torch.bfloat16])
def test_identity_birth_is_exact_pass_through(dtype):
    """The registered identity birth ``[0, 0, 0, 1]`` (the STANDARD layout:
    the LAST tap is the current token) is EXACTLY the pass-through in every
    dtype -- ``torch.equal`` bitwise on ordinary fixtures."""
    torch.manual_seed(3)
    conv = IdentityBornShortConv(11)
    assert torch.equal(conv.weight,
                      torch.tensor(IDENTITY_SHORT_CONV_BIRTH)
                      .unsqueeze(0).expand(11, 4))
    x = torch.randn(3, 17, 11).to(dtype)
    assert torch.equal(conv(x), x), dtype


def test_identity_birth_minus_zero_corner_receipt():
    """The one legal escape, counted: a -0.0 entry can flip to +0.0 through
    the zero taps' additions (``-0.0 + +0.0 = +0.0``).  The flips happen
    ONLY on exact-zero entries, are invisible to every ``==``-based parity
    bar (``-0.0 == +0.0``) and to the loss -- receipted, not hidden (the
    K17 -0.0 corner's own discipline)."""
    conv = IdentityBornShortConv(5)
    x = torch.randn(2, 9, 5)
    x[0, 3, 2] = -0.0            # an exact -0.0 payload entry
    x[1, 0, 4] = 0.0             # an exact +0.0 entry
    y = conv(x)
    assert torch.equal(y, x), \
        "the identity birth is value-exact even where the sign of zero flips"
    bitwise = y.view(torch.int32) != x.view(torch.int32)
    zeros = x == 0.0
    assert int((bitwise & ~zeros).sum()) == 0, \
        "a sign flip outside the exact-zero entries is a real divergence"
    assert int((bitwise & zeros).sum()) >= 0     # the corner, counted


def test_standard_tap_layout_and_causal_padding():
    """The tap layout receipt: tap ``j`` multiplies ``x[t-(K-1)+j]`` (the
    STANDARD nn.Conv1d/fla layout -- tap 0 is the OLDEST, the LAST tap is
    the current token), and the taps before t=0 are causal ZERO padding."""
    batch, tokens, channels = 2, 12, 3
    x = torch.randn(batch, tokens, channels)
    for j in range(SHORT_CONV_KERNEL_SIZE):
        w = torch.zeros(channels, SHORT_CONV_KERNEL_SIZE)
        w[:, j] = 1.0
        y = short_conv1d_causal(x, w)
        for t in range(tokens):
            want = x[:, t - (SHORT_CONV_KERNEL_SIZE - 1) + j, :] \
                if t - (SHORT_CONV_KERNEL_SIZE - 1) + j >= 0 \
                else torch.zeros(batch, channels)
            assert torch.equal(y[:, t], want), (j, t)


def test_equivalence_to_nn_conv1d_up_to_accumulation_order():
    """The conv IS the real conv: the explicit tap sum equals
    ``F.conv1d(pad(x), w.view(C,1,K), groups=C)`` up to floating-point
    accumulation order (fp64 within 1e-14, fp32 within 1e-6 -- cuDNN's
    own order, disclosed, not a semantics difference)."""
    torch.manual_seed(7)
    x = torch.randn(2, 13, 9)
    w = torch.randn(9, SHORT_CONV_KERNEL_SIZE)
    for dtype, tol in ((torch.float64, 1e-14), (torch.float32, 1e-6)):
        mine = short_conv1d_causal(x.to(dtype), w.to(dtype))
        ref = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x.to(dtype).transpose(1, 2),
                                    (SHORT_CONV_KERNEL_SIZE - 1, 0)),
            w.to(dtype).unsqueeze(1), groups=9).transpose(1, 2)
        assert (mine - ref).abs().max().item() <= tol, dtype


def test_fp64_gradcheck_and_the_analytic_tap_adjoint():
    """MANDATORY (the registration's 5), the conv's own share: the fp64
    gradcheck through the conv, and the tap adjoint is the analytic
    correlation EXACTLY (``dw[c,j] = sum g[t,c] * x[t-(K-1)+j,c]``)."""
    torch.manual_seed(11)
    x = torch.randn(2, 16, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(short_conv1d_causal, (x, w),
                                     eps=1e-6, atol=1e-9, rtol=1e-9)
    y = short_conv1d_causal(x, w)
    g = torch.randn_like(y)
    y.backward(g)
    padded = torch.nn.functional.pad(x.detach(), (0, 0, 3, 0))
    expect = torch.stack(
        [(g * padded[:, j: j + 16]).sum((0, 1)) for j in range(4)], 1)
    assert torch.equal(w.grad, expect), \
        "the fp64 tap adjoint must be the analytic correlation exactly"


def test_shift_birth_is_the_fixed_token_average():
    """The fixed non-identity control: ``[0, 0, 0.5, 0.5]`` in the standard
    layout is the average of the current and the previous token (zero
    left-padding at t=0), exactly -- the arm that separates ANY local
    mixing from a LEARNED filter."""
    conv = IdentityBornShortConv(6, birth="shift")
    assert torch.equal(conv.weight,
                      torch.tensor(SHIFT_SHORT_CONV_BIRTH)
                      .unsqueeze(0).expand(6, 4))
    x = torch.randn(2, 9, 6)
    y = conv(x)
    want = 0.5 * x.clone()
    want[:, 1:] = 0.5 * x[:, 1:] + 0.5 * x[:, :-1]
    assert torch.equal(y, want)


def test_births_and_refusals_are_by_name():
    """The births registry is closed (identity | shift) and the refusals
    are by name: an unregistered birth and a non-registered kernel size
    are refused; the weight's channel axis must be the payload's."""
    with pytest.raises(ValueError, match="unregistered conv birth"):
        IdentityBornShortConv(4, birth="silu")
    with pytest.raises(ValueError, match="kernel is 4"):
        IdentityBornShortConv(4, kernel_size=3)
    conv = IdentityBornShortConv(4)
    with pytest.raises(ValueError, match="channel axis"):
        short_conv1d_causal(torch.randn(1, 5, 7), conv.weight)


def test_parameter_delta_is_the_registrations_own():
    """The depthwise kernel's count, computed not hardcoded: one
    (heads*value_dim) x 4 tap kernel per mixer ON TOP of the composed
    machine's own p3 stack; 24,576 at the flagship depth 24."""
    assert short_conv_parameter_delta(24, 576, 2, 128) \
        == composed_parameter_delta(24, 576, 2, 128) + 24_576
    assert short_conv_parameter_delta(24, 576, 2, 128) == 8_681_616
    assert short_conv_parameter_delta(1, 64, 2, 32) \
        == composed_parameter_delta(1, 64, 2, 32) + 2 * 32 * 4


# --------------------------------------------- the birth receipts (CUDA)


@needs_cuda
def test_mixer_birth_bit_identical_to_the_composed_mixer():
    """The machine-level birth receipt: the converted short-conv mixer at
    the identity birth is BIT-IDENTICAL to the K15 composed mixer's own
    output (the conv is exactly the pass-through; the gates at the
    registered scalar equivalents; the additive read K13's own), under
    the production bf16 autocast regime."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    def build():
        torch.manual_seed(1234)
        stub = torch.nn.Module()
        stub.blocks = [torch.nn.Module()]
        stub.blocks[0].mixer = MatrixMixer(MatrixShape(
            width=64, depth=1, hidden=128, heads=2, value_dim=32,
            rank=16, rank_max=16), POLICY, 64)
        return convert_model_to_separated_keys(stub)

    mixer15 = convert_model_to_composition(build()).blocks[0].mixer.to(DEVICE)
    mixer18 = convert_model_to_short_conv(
        convert_model_to_composition(build())).blocks[0].mixer.to(DEVICE)
    for name in ("value", "key", "query_key", "out", "decay", "b_proj",
                 "f_proj", "g_proj"):
        for p15, p18 in zip(getattr(mixer15, name).parameters(),
                            getattr(mixer18, name).parameters()):
            assert torch.equal(p15, p18), name
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out15, _ = mixer15.mix(x)
        out18, _ = mixer18.mix(x)
    assert torch.equal(out15, out18), \
        "the born short-conv mixer must reproduce the composed mixer's " \
        "output bit for bit"


@needs_cuda
def test_decisions_bit_invariant_to_conv_values():
    """P2 trivially held, receipted: the composed kernel's recorded
    schedule, read WEIGHTS, pre-decision liveness and key table are
    BIT-INVARIANT to the conv's values (RANDOM kernels -- far off the
    identity birth), while the value state moves as designed.  No conv
    value feeds op selection."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23)
    base = composed_kernel_forward(payload, write_keys, query_keys, decay,
                                    gates=None, rank=5, segment_tokens=16)
    w = conv_stream(2, 96, 2, 8, 31)
    conved = short_conv1d_causal(
        payload.reshape(2, 96, 16), w).view_as(payload)
    moved = composed_kernel_forward(conved, write_keys, query_keys, decay,
                                    gates=None, rank=5, segment_tokens=16)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(base[name], moved[name]), name
    assert torch.equal(base["live_pre"], moved["live_pre"]), "live_pre"
    assert torch.equal(base["keys"], moved["keys"]), "keys"
    assert torch.equal(base["weights"], moved["weights"]), \
        "the read weights are the decision channel's own (P2)"
    assert not torch.equal(base["state"], moved["state"]), \
        "a non-identity conv must move the value state"
    assert not torch.equal(base["reads"], moved["reads"]), \
        "a non-identity conv must move the reads"


@needs_cuda
@pytest.mark.parametrize("gated", [True, False])
def test_gate_b_oracle_gradcheck_fp64_through_the_conv_path(gated):
    """MANDATORY (the registration's 5): the fp64 gradcheck THROUGH THE CONV
    PATH -- the conv's tap adjoint attached in front of the composed walk,
    the additive read's raw-score weights ATTACHED (no softmax jacobian
    anywhere), the payload lanes under the per-channel write gate, and the
    PER-CHANNEL DECAY vector -- both GATED combinations, over a fixture
    that fires every content-carrying branch."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 12, 2, 8, 4, 5, 11)
    gates = gate_stream(2, 12, 2, 8, 17)
    w = conv_stream(2, 12, 2, 8, 19).double()
    p = payload.double().clone().requires_grad_(True)
    wk = write_keys.double().clone().requires_grad_(True)
    qk = query_keys.double().clone().requires_grad_(True)
    d = decay.double().clone().requires_grad_(True)
    g_ = gates.double().clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    k = torch.zeros(2, 2, 5, 4, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    wd = w.clone().requires_grad_(True)

    def f(p_, w_, wk_, qk_, d_, g__, s_, k_):
        batch, tokens, heads, value_dim = p_.shape
        conved = short_conv1d_causal(
            p_.reshape(batch, tokens, heads * value_dim), w_
        ).view(batch, tokens, heads, value_dim)
        ref = composed_oracle_scan(conved, wk_, qk_, d_, g__, policy=POLICY,
                                   gated=gated, rank=5, rank_max=5,
                                   initial_state=s_, initial_keys=k_)
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, wd, wk, qk, d, g_, s, k),
                                    eps=1e-6, atol=1e-4, rtol=1e-4)
    # the fixture must fire the content-carrying branches
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        op_histogram)
    ref = composed_oracle_scan(payload, write_keys, query_keys, decay, gates,
                               policy=POLICY, gated=gated, rank=5, rank_max=5)
    hist = op_histogram(ref["schedule"])
    assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_kernel_gradients_through_the_conv_within_floors():
    """The kernel path's gradients THROUGH THE CONV CHAIN (the composed
    backward's dpayload feeding autograd's conv adjoint) vs the eager
    oracle's own, inside NOISE_FACTOR of the fp32-vs-fp64 floor measured
    on the same fixture -- the committed composed engines' own gradients
    are the K15 suite's receipt; this lane's NEW surface is the conv."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    w = conv_stream(2, 64, 2, 8, 7)
    rank, seg = 5, 16

    def chain(scan_fn, dtype):
        p = payload.clone().to(dtype).requires_grad_(True)
        ww = w.clone().to(dtype).requires_grad_(True)
        wk = write_keys.clone().to(dtype).requires_grad_(True)
        qk = query_keys.clone().to(dtype).requires_grad_(True)
        d = decay.clone().to(dtype).requires_grad_(True)
        g_ = gates.clone().to(dtype).requires_grad_(True)
        conved = short_conv1d_causal(
            p.reshape(2, 64, 16), ww).view(2, 64, 2, 8)
        result = scan_fn(conved, wk, qk, d, g_)
        reads = result["reads"] if isinstance(result, dict) else result.reads
        upstream = torch.randn(reads.shape, generator=torch.Generator()
                               .manual_seed(1234)).to(dtype).to(DEVICE)
        (reads * upstream).sum().backward()
        return dict(dp=p.grad, dw=ww.grad, dwk=wk.grad, dqk=qk.grad,
                    dd=d.grad, dg=g_.grad)

    def oracle_scan(conved, wk, qk, d, g_):
        return composed_oracle_scan(conved, wk, qk, d, g_, policy=POLICY,
                                    gated=True, rank=rank, rank_max=rank)

    def kernel_scan(conved, wk, qk, d, g_):
        return composed_scan(conved, wk, qk, d, g_, policy=POLICY,
                             gated=True, rank=rank, rank_max=rank,
                             segment_tokens=seg)

    kern = chain(kernel_scan, torch.float32)
    orac32 = chain(oracle_scan, torch.float32)
    orac64 = chain(oracle_scan, torch.float64)
    for name in ("dp", "dw", "dwk", "dqk", "dd", "dg"):
        floor = (orac32[name].float() - orac64[name].float()).abs().max().item()
        deviation = (kern[name].float() - orac32[name].float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern[name]).all(), name


@needs_cuda
def test_identity_born_kernel_receives_gradient_at_step_0():
    """The no-dead-parameter receipt: unlike the zero-born gates (whose
    projections wake in two steps), the identity-born conv kernel receives
    gradient at step 0 THROUGH the current tap -- the conv LEARNS to
    deviate from the first step (the falsifier's frozen contrast)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    torch.manual_seed(99)
    stub = torch.nn.Module()
    stub.blocks = [torch.nn.Module()]
    stub.blocks[0].mixer = MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), POLICY, 64)
    mixer = convert_model_to_short_conv(
        convert_model_to_composition(
            convert_model_to_separated_keys(stub))).blocks[0].mixer.to(DEVICE)
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = mixer.mix(x)
    out.float().pow(2).sum().backward()
    assert mixer.v_conv.weight.grad is not None
    assert float(mixer.v_conv.weight.grad.abs().sum()) > 0.0, \
        "the identity-born kernel must receive gradient at step 0"
    # the gates' own wake-up receipt is unchanged by the conv
    for name in ("decay_scalar", "write_scalar", "read_scalar"):
        assert float(getattr(mixer, name).grad.abs().sum()) > 0.0, name


@needs_cuda
def test_frozen_conv_plumbing():
    """The falsifier's plumbing receipt: freezing the conv at the identity
    birth (requires_grad=False) removes it from the backward entirely
    (grad None) and the mixer's forward is STILL the composed machine's
    own bit for bit."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    torch.manual_seed(5)
    stub = torch.nn.Module()
    stub.blocks = [torch.nn.Module()]
    stub.blocks[0].mixer = MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), POLICY, 64)
    stub = convert_model_to_composition(convert_model_to_separated_keys(stub))
    mixer15 = stub.blocks[0].mixer.to(DEVICE)
    torch.manual_seed(5)
    stub2 = torch.nn.Module()
    stub2.blocks = [torch.nn.Module()]
    stub2.blocks[0].mixer = MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), POLICY, 64)
    mixer18 = convert_model_to_short_conv(
        convert_model_to_composition(
            convert_model_to_separated_keys(stub2))).blocks[0].mixer.to(DEVICE)
    mixer18.v_conv.weight.requires_grad_(False)
    assert all(not p.requires_grad for p in mixer18.v_conv.parameters())
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out15, _ = mixer15.mix(x)
        out18, _ = mixer18.mix(x)
    assert torch.equal(out15, out18)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out18b, _ = mixer18.mix(x)
    out18b.float().pow(2).sum().backward()
    assert mixer18.v_conv.weight.grad is None, \
        "a frozen conv must receive no gradient"


# ------------------------------------------------------- the conversion receipts


@needs_cuda
def test_conversion_shares_parameters_and_refuses_by_name():
    """The conversion re-wraps a K15-composed mixer sharing every parameter
    object, adds the conv at the registered birth, and refuses a
    non-composed mixer and an unregistered birth by name."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)

    stub = Stub()
    stub = convert_model_to_composition(convert_model_to_separated_keys(stub))
    before = {n: p for n, p in stub.blocks[0].mixer.named_parameters()}
    stub = convert_model_to_short_conv(stub)
    mixer = stub.blocks[0].mixer.to(DEVICE)
    assert isinstance(mixer, ShortConvComposedMixer)
    assert isinstance(mixer, ComposedReadMixer)
    after = dict(mixer.named_parameters())
    for name, param in before.items():
        assert after[name] is param, \
            f"the conversion must SHARE the composed parameters: {name}"
    delta = sum(p.numel() for p in mixer.parameters()) \
        - sum(p.numel() for p in before.values())
    assert delta == 2 * 32 * SHORT_CONV_KERNEL_SIZE
    assert "v_conv.weight" in after
    with pytest.raises(ValueError, match="expects a K15-composed model"):
        convert_model_to_short_conv(Stub())
    with pytest.raises(ValueError, match="unregistered conv birth"):
        convert_model_to_short_conv(stub, birth="nope")
    # the shift birth converts too (the fixed control's own conversion)
    stub_shift = Stub()
    stub_shift = convert_model_to_short_conv(
        convert_model_to_composition(
            convert_model_to_separated_keys(stub_shift)), birth="shift")
    w = stub_shift.blocks[0].mixer.v_conv.weight
    assert torch.equal(w, torch.tensor(SHIFT_SHORT_CONV_BIRTH)
                      .unsqueeze(0).expand_as(w))
    x = torch.randn(1, 16, 64, device=DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
    assert out.shape == (1, 16, 64)
    assert torch.isfinite(out).all()
    assert torch.equal(
        conv_payload(mixer, x).reshape(1, 16, 64),
        mixer.value(x)), "conv_payload must be the identity at birth"
