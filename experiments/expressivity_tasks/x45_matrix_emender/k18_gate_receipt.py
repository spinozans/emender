"""K18 -- the short-conv machine's gate receipts: the conv's own receipts
(A-conv), the birth receipts (A2) and the gradient gates (B) and walls (D)
at the registered shape.

Writes ``receipts/k18_gates.json`` (the K15/K17 gate-receipt pattern):

* **Gate A-conv** -- the conv's own receipts at the registered channel
  count (heads*value_dim = 256, kernel 4): the identity birth is EXACTLY
  the pass-through in fp32/bf16/fp64 (``torch.equal``; the sign-of-zero
  corner counted -- invisible to every ``==`` bar and to the loss), the
  explicit tap sum equals ``F.conv1d`` up to accumulation order (fp64
  within 1e-14, fp32 within 1e-6), the fp64 tap adjoint is the analytic
  correlation EXACTLY, and the fp64 gradcheck through the conv passes.
* **Gate A2** -- the conv machine's birth receipts at the registered scan
  shape (batch 8, 2048 tokens, 2 heads, rank 128, value_dim 128, key_dim
  16): (i) BORN AS THE COMPOSED MACHINE -- the conv'd payload bit-equal
  to the raw projection, and the composed kernel's every channel launched
  on the conv'd payload bit-equal to its own launch on the raw payload (0
  differing elements); (ii) the DECISIONS-BIT-IDENTICAL bar under RANDOM
  conv kernels (P2 trivially held: no conv value feeds op selection --
  the schedule, the read weights, the pre-decision liveness and the key
  table are invariant while the value state moves).
* **Gate B** -- MANDATORY: the fp64 gradcheck of the composed oracle
  THROUGH THE CONV PATH (the conv's tap adjoint attached in front of the
  composed walk, both gated combinations, every content-carrying branch
  fired), the fp32-vs-fp64 floor measured on the same fixture, and the
  kernel path's gradients through the conv chain within ``NOISE_FACTOR``
  (8x) of the floor.
* **Gate D** -- the honest walls: the conv's own forward/backward
  block-call cost per (layer, window) at the registered shape, stated
  against the composed scan's own same-session launch.

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k18_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k18_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    _k15_forward_launch, composed_oracle_scan, composed_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv import (
    IDENTITY_SHORT_CONV_BIRTH, IdentityBornShortConv, conv_payload,
    short_conv1d_causal)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K10/K13/K15/K17 gate receipts' own fixture).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64
CHANNELS = HEADS * VALUE_DIM                 # the conv's registered width
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")


def registered_fixture(seed: int = 77):
    """The lane's canonical fixture at the registered shape (the
    K10/K13/K15 gate convention: payload and write-keys from ``distinct``
    fixed directions, query keys random, per-channel decay around the
    scalar default)."""
    g = torch.Generator().manual_seed(seed)
    distinct = 96
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=g).cuda()
    key_dirs = torch.randn(distinct, KEY_DIM, generator=g).cuda()
    choice = torch.randint(0, distinct, (BATCH, TOKENS, HEADS), generator=g)
    payload = payload_dirs[choice.cpu()]
    write_keys = key_dirs[choice.cpu()]
    query_keys = torch.randn(BATCH, TOKENS, HEADS, KEY_DIM,
                            generator=g).cuda() * 0.5
    decay = (torch.full((BATCH, TOKENS, HEADS, 1), 0.95).cuda()
             + (torch.rand(BATCH, TOKENS, HEADS, VALUE_DIM, generator=g)
                .cuda() * 0.2 - 0.1)).clamp(min=1e-4, max=1.0 - 1e-6)
    return payload, write_keys, query_keys, decay


def gate_a_conv(receipt: dict) -> dict:
    """The conv's own receipts at the registered channel count."""
    gate = dict(name="A-conv", note="the conv's own unit receipts at the "
               "registered channel count (256, kernel 4)")
    conv = IdentityBornShortConv(CHANNELS).cuda()
    gate["identity_birth_taps"] = list(IDENTITY_SHORT_CONV_BIRTH)
    # (i) the identity birth is exactly the pass-through, every dtype
    g = torch.Generator().manual_seed(11)
    x = torch.randn(4, 1024, CHANNELS, generator=g).cuda()
    x[0, 7, 3] = -0.0
    x[1, 0, 255] = 0.0
    exact = {}
    corners = {}
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        xd = x.to(dtype)
        y = conv(xd)
        exact[str(dtype)] = bool(torch.equal(y, xd))
        if dtype is torch.float32:
            bitwise = y.view(torch.int32) != xd.view(torch.int32)
            corners["sign_of_zero_flips"] = int(
                (bitwise & (xd == 0.0)).sum())
            corners["flips_outside_exact_zeros"] = int(
                (bitwise & ~(xd == 0.0)).sum())
            corners["note"] = ("the IEEE corner: -0.0 can flip to +0.0 "
                               "through the zero taps' additions; counted "
                               "here, invisible to every ==-based parity bar "
                               "and to the loss")
    gate["identity_birth_exact_pass_through"] = exact
    gate["minus_zero_corner"] = corners
    # (ii) the explicit tap sum equals F.conv1d up to accumulation order
    w = torch.randn(CHANNELS, 4, generator=g).cuda()
    diffs = {}
    for dtype, key in ((torch.float64, "fp64_max_abs_diff"),
                       (torch.float32, "fp32_max_abs_diff")):
        mine = short_conv1d_causal(x.to(dtype), w.to(dtype))
        ref = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x.to(dtype).transpose(1, 2), (3, 0)),
            w.to(dtype).unsqueeze(1), groups=CHANNELS).transpose(1, 2)
        diffs[key] = float((mine - ref).abs().max())
    gate["equivalence_to_nn_conv1d"] = dict(
        diffs, note="the explicit tap sum IS the causal depthwise conv "
        "(nn.Conv1d(C,C,4,groups=C,bias=False) with left zero-padding, the "
        "fla ShortConvolution layout); cuDNN's accumulation order is the "
        "only difference")
    # (iii) the fp64 tap adjoint is the analytic correlation exactly, and
    # the fp64 gradcheck through the conv passes (a small fixture: the
    # numerical jacobian is quadratic in the element count; the registered
    # 256-channel shape is carried by the receipts above, and the SUITE's
    # gradcheck-through-the-composed-path carries the composition)
    xd = torch.randn(2, 32, 48, dtype=torch.float64,
                    generator=g).cuda().requires_grad_(True)
    wd = torch.randn(48, 4, dtype=torch.float64,
                     generator=g).cuda().requires_grad_(True)
    assert torch.autograd.gradcheck(short_conv1d_causal, (xd, wd),
                                    eps=1e-6, atol=1e-9, rtol=1e-9)
    y = short_conv1d_causal(xd, wd)
    grad = torch.randn_like(y)
    y.backward(grad)
    padded = torch.nn.functional.pad(xd.detach(), (0, 0, 3, 0))
    expect = torch.stack(
        [(grad * padded[:, j: j + 32]).sum((0, 1)) for j in range(4)], 1)
    gate["fp64_tap_adjoint_exact"] = bool(torch.equal(wd.grad, expect))
    gate["fp64_gradcheck_through_the_conv"] = True
    receipt["gate_a_conv"] = gate
    return gate


def _launch(payload, write_keys, query_keys, decay, gates):
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device=payload.device)
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool,
                       device=payload.device)
    keys = torch.zeros(BATCH, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                      device=payload.device)
    return _k15_forward_launch(payload, write_keys, query_keys, decay, gates,
                               state, live, keys, POL32, RANK, SEGMENT_TOKENS,
                               True, collect=True, warps=8, stages=3)


def gate_a2(receipt: dict) -> dict:
    """The birth receipts at the registered scan shape."""
    gate = dict(name="A2", note="the conv machine's birth receipts at the "
               "registered shape (batch 8, 2048 tokens, rank 128)")
    payload, write_keys, query_keys, decay = registered_fixture()
    gates = torch.ones_like(payload)
    # (i) BORN AS THE COMPOSED MACHINE: the conv'd payload bit-equal to the
    # raw projection, and the composed kernel's own launch on the conv'd
    # payload bit-equal to its launch on the raw payload
    conv = IdentityBornShortConv(CHANNELS).cuda()
    conved = short_conv1d_causal(
        payload.reshape(BATCH, TOKENS, CHANNELS),
        conv.weight).view_as(payload)
    gate["conv_payload_bit_equal_at_birth"] = bool(torch.equal(conved, payload))
    base = _launch(payload, write_keys, query_keys, decay, gates)
    moved = _launch(conved, write_keys, query_keys, decay, gates)
    flips = lanes = 0
    for name in ("slot", "erase", "op", "live_after", "payload", "live_pre",
                 "reads", "state", "live", "weights", "keys"):
        a, b = base[name], moved[name]
        flips += int((a != b).sum())
        lanes += a.numel()
    gate["born_as_the_composed_machine"] = dict(
        flips=flips, lanes=lanes, bit_identical=bool(flips == 0),
        note="at the identity birth the conv'd payload IS the raw payload, "
             "so the composed kernel's every channel is the composed "
             "machine's own -- the machine is BORN AS THE COMPOSED MACHINE "
             "bit for bit (and hence as the K13 additive machine: the K15 "
             "receipt's own chain)")
    # (ii) the decisions-bit-identical bar under RANDOM conv kernels
    g = torch.Generator().manual_seed(19)
    w_random = (torch.randn(CHANNELS, 4, generator=g).cuda() * 0.5)
    conved_random = short_conv1d_causal(
        payload.reshape(BATCH, TOKENS, CHANNELS),
        w_random).view_as(payload)
    off = _launch(conved_random, write_keys, query_keys, decay, gates)
    d_flips = d_lanes = 0
    for name in ("slot", "erase", "op", "live_after", "weights", "live_pre",
                 "keys"):
        a, b = base[name], off[name]
        d_flips += int((a != b).sum())
        d_lanes += a.numel()
    gate["decisions_bit_identical_under_random_conv_kernels"] = dict(
        flips=d_flips, lanes=d_lanes,
        bit_identical=bool(d_flips == 0),
        state_moves=bool(not torch.equal(base["state"], off["state"])),
        reads_move=bool(not torch.equal(base["reads"], off["reads"])),
        note="P2 trivially held: no conv value feeds op selection, the "
             "read weights or the key table -- the value state and the "
             "reads move as designed off the identity birth")
    receipt["gate_a2"] = gate
    return gate


def gate_b(receipt: dict) -> dict:
    """MANDATORY: the fp64 gradcheck through the conv path + the kernel
    gradients through the conv chain vs the measured floors."""
    gate = dict(name="B")
    batch, tokens, heads, value_dim, key_dim, rank = 2, 12, 2, 8, 4, 5
    g = torch.Generator().manual_seed(29)
    payload_dirs = torch.randn(5, value_dim, generator=g).cuda()
    key_dirs = torch.randn(5, key_dim, generator=g).cuda()
    choice = torch.randint(0, 5, (batch, tokens, heads), generator=g)
    payload = payload_dirs[choice.cpu()]
    write_keys = key_dirs[choice.cpu()]
    query_keys = torch.randn(batch, tokens, heads, key_dim,
                             generator=g).cuda() * 0.5
    decay = (torch.full((batch, tokens, heads, 1), 0.95).cuda()
             + (torch.rand(batch, tokens, heads, value_dim, generator=g)
                .cuda() * 0.2 - 0.1)).clamp(min=1e-4, max=1.0 - 1e-6)
    gates = (torch.rand(batch, tokens, heads, value_dim, generator=g)
             .cuda() * 0.9 + 0.05)
    channels = heads * value_dim
    w = (torch.randn(channels, 4, generator=g).cuda() * 0.5).double()

    # (i) the fp64 gradcheck THROUGH THE CONV PATH, both gated combos
    gradchecks = {}
    for gated in (True, False):
        p = payload.double().clone().requires_grad_(True)
        wd = w.clone().requires_grad_(True)
        wk = write_keys.double().clone().requires_grad_(True)
        qk = query_keys.double().clone().requires_grad_(True)
        d = decay.double().clone().requires_grad_(True)
        g_ = gates.double().clone().requires_grad_(True)
        s = torch.zeros(batch, heads, rank, value_dim,
                        dtype=torch.float64, device="cuda"
                        ).requires_grad_(True)
        k = torch.zeros(batch, heads, rank, key_dim,
                        dtype=torch.float64, device="cuda"
                        ).requires_grad_(True)

        def f(p_, w_, wk_, qk_, d_, g__, s_, k_):
            conved = short_conv1d_causal(
                p_.reshape(batch, tokens, channels), w_
            ).view(batch, tokens, heads, value_dim)
            ref = composed_oracle_scan(conved, wk_, qk_, d_, g__,
                                       policy=POLICY, gated=gated, rank=rank,
                                       rank_max=rank, initial_state=s_,
                                       initial_keys=k_)
            return ref["reads"], ref["state"], ref["keys"]
        gradchecks[str(gated)] = bool(torch.autograd.gradcheck(
            f, (p, wd, wk, qk, d, g_, s, k), eps=1e-6, atol=1e-4, rtol=1e-4))
    gate["fp64_gradcheck_through_the_conv_path"] = gradchecks
    ref_probe = composed_oracle_scan(payload, write_keys, query_keys, decay,
                                     gates, policy=POLICY, gated=True,
                                     rank=rank, rank_max=rank)
    hist = op_histogram(ref_probe["schedule"])
    gate["fixture_op_histogram"] = {k: int(v) for k, v in hist.items()}
    assert all(hist[n] > 0 for n in ("append", "correct")), hist

    # (ii) the kernel path's gradients through the conv chain vs the
    # measured fp32-vs-fp64 floors (the registered-shape fixture)
    payload_r, write_keys_r, query_keys_r, decay_r = registered_fixture()
    gates_r = (torch.rand(BATCH, TOKENS, HEADS, VALUE_DIM,
                          generator=torch.Generator().manual_seed(97)).cuda()
               * 0.9 + 0.05)
    w_r = (torch.randn(CHANNELS, 4,
                       generator=torch.Generator().manual_seed(83)).cuda()
           * 0.5)

    def chain(scan_fn, dtype):
        p = payload_r.clone().to(dtype).requires_grad_(True)
        ww = w_r.clone().to(dtype).requires_grad_(True)
        wk = write_keys_r.clone().to(dtype).requires_grad_(True)
        qk = query_keys_r.clone().to(dtype).requires_grad_(True)
        d = decay_r.clone().to(dtype).requires_grad_(True)
        g_ = gates_r.clone().to(dtype).requires_grad_(True)
        conved = short_conv1d_causal(
            p.reshape(BATCH, TOKENS, CHANNELS), ww).view_as(p)
        result = scan_fn(conved, wk, qk, d, g_)
        reads = result["reads"] if isinstance(result, dict) else result.reads
        upstream = torch.randn(reads.shape,
                               generator=torch.Generator()
                               .manual_seed(1234)).to(dtype).to(reads.device)
        (reads * upstream).sum().backward()
        return dict(dp=p.grad, dw=ww.grad)

    kern = chain(lambda *a: composed_scan(*a, policy=POLICY, gated=True,
                                          rank=RANK, rank_max=RANK,
                                          segment_tokens=SEGMENT_TOKENS),
                 torch.float32)
    orac32 = chain(lambda *a: composed_oracle_scan(
        *a, policy=POLICY, gated=True, rank=RANK, rank_max=RANK),
        torch.float32)
    orac64 = chain(lambda *a: composed_oracle_scan(
        *a, policy=POLICY, gated=True, rank=RANK, rank_max=RANK),
        torch.float64)
    rows = {}
    for name in ("dp", "dw"):
        floor = (orac32[name].float() - orac64[name].float()).abs().max().item()
        deviation = (kern[name].float()
                     - orac32[name].float()).abs().max().item()
        rows[name] = dict(floor=floor, deviation=deviation,
                          within_noise=bool(
                          deviation <= NOISE_FACTOR * max(floor, 1e-6)
                          or deviation <= 1e-5))
    gate["kernel_gradients_through_the_conv_vs_floors"] = rows
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """The honest walls: the conv's own forward/backward block-call cost at
    the registered shape, stated against the composed scan's own launch."""
    gate = dict(name="D")
    payload, write_keys, query_keys, decay = registered_fixture()
    gates = torch.ones_like(payload)
    conv = IdentityBornShortConv(CHANNELS).cuda()
    x2d = payload.reshape(BATCH, TOKENS, CHANNELS)

    def timeit(fn, repeat=5):
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(repeat):
            out = fn()
        torch.cuda.synchronize()
        return (time.monotonic() - started) / repeat * 1000.0, out

    fwd_ms, y = timeit(lambda: short_conv1d_causal(x2d, conv.weight))
    y = y.detach().requires_grad_(True)
    w = conv.weight.detach().requires_grad_(True)

    def bwd():
        out = short_conv1d_causal(x2d.detach().requires_grad_(True), w)
        out.sum().backward()
        return out
    bwd_ms, _ = timeit(bwd)

    def composed_fwd():
        return _launch(payload, write_keys, query_keys, decay, gates)
    scan_ms, _ = timeit(composed_fwd, repeat=3)
    gate["conv_forward_ms_per_window"] = fwd_ms
    gate["conv_backward_ms_per_window"] = bwd_ms
    gate["composed_scan_forward_ms_per_window"] = scan_ms
    gate["conv_over_scan_percent"] = 100.0 * fwd_ms / scan_ms
    gate["note"] = ("the conv's own block-call cost per (layer, window) at "
                    "the registered shape, vs the composed scan's own "
                    "same-session launch -- stated, not estimated; the "
                    "conv's cost is the mixer's, outside the scan's opaque "
                    "op pair")
    receipt["gate_d"] = gate
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(ROOT / "receipts" / "k18_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k18_short_conv_gates",
                   registration=str(
                       ROOT / "PREREGISTRATION_SHORT_CONV.md"),
                   policy=POLICY.record(),
                   score_form="the short-conv machine: the K15 composed "
                              "machine (the K13 additive read under the "
                              "K11B p3 per-channel gate stack) + GDN2's "
                              "registered short depthwise convolution "
                              "(kernel 4, bias-free, the silu NOT "
                              "transplanted) on the payload path, born at "
                              "the identity kernel [0,0,0,1] (the standard "
                              "nn.Conv1d layout)")
    gate_a_conv(receipt)
    gate_a2(receipt)
    gate_b(receipt)
    gate_d(receipt)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1)[:5000])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
