"""K22 -- THE HYBRID's gate receipts: the birth discipline, at the seam.

The lane's registered receipts, run on one leased card BEFORE any arm
(``receipts/k22_gates.json``):

* **Gate A -- BORN AS GDN2, bit for bit.**  The hybrid (the committed GDN2
  build + the K22 graft at the slot layers, the read gates born at exact
  0.0) against a pure-GDN2 twin of the SAME build: one real production
  microbatch (the flagship stream's own first microbatch group), the full
  model loss asserted bit-identical, and every slot layer's combined
  output asserted bit-identical to the pure layer's own -- flips counted
  bitwise (the only legal escape is the IEEE sign-of-zero corner: a +0.0
  slot contribution added to a -0.0 dense entry; counted separately,
  invisible to every value comparison and to the loss).
* **Gate A2 -- the seam's dtype discipline.**  Under the production bf16
  autocast regime the wrapper's output dtype is the dense layer's own
  (the contribution is cast to the dense output's carrier dtype), so the
  residual stream's dtype is the committed machine's own at birth.
* **Gate B -- the fp64 gradcheck THROUGH THE HYBRID SEAM** (the
  slot-in-dense-stack boundary): ``dense_out + gate * slot(x)`` with the
  slot path walked by the composed ORACLE (the committed eager parity
  path, fp64-capable) -- gradcheck over the dense leaf, the gate, the
  input and every slot parameter, both gated combinations; then the
  KERNEL path's seam gradients (through the committed ``composed_scan``
  opaque op pair) within the measured fp32-vs-fp64 floors x
  ``NOISE_FACTOR``.
* **Gate C -- the open signal and the inert birth.**  At the zero-born
  gate: ``dLoss/dgate`` is NONZERO (the open signal exists -- the gate's
  gradient is ``<upstream, slot output>``), while every OTHER slot
  parameter's gradient is EXACTLY ZERO (the gate's zero cuts the
  gradient); and the slot stores FILL on the same forward (the machinery
  runs, nothing reads out).
* **Gate D -- the honest wall.**  The slot branch's own forward cost per
  (slot layer, window-group) at the registered production shape, stated
  against the K20/K21 receipts (the committed composed mixer's own
  measured cost).

Usage (one leased card, cards 2/3):

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \\
    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k22_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k22_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    composed_oracle_scan, composed_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv import (
    short_conv1d_causal)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
    import _round_f32
from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid import (
    SLOT_LAYERS, HybridSlotGDN2Layer, build_slot_mixer, slot_policy)
from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
    import causal_context

ROOT = Path(__file__).resolve().parent
NOISE_FACTOR = 8.0
SEED = 3401


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's raw bit pattern as an integer view (the bitwise
    comparison -- ``-0.0`` vs ``+0.0`` compare equal by value but not by
    bits)."""
    return tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 \
        else tensor.view(torch.int32)


def _flip_rows(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bitwise flip accounting between two value-equal-expected tensors:
    total flips, flips on exact-zero entries (the IEEE sign-of-zero
    corner: legal, invisible to every value bar), and flips OUTSIDE
    exact-zero entries (must be ZERO -- a real divergence)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(shape_mismatch=(list(a.shape), list(b.shape)),
                    dtype_mismatch=(str(a.dtype), str(b.dtype)),
                    flips_outside_exact_zero=-1)
    diff = _bits(a.detach()) != _bits(b.detach())
    zero = (a.detach() == 0) | (b.detach() == 0)
    corner = bool((diff & zero).sum().item())
    outside = bool((diff & ~zero).sum().item())
    return dict(lanes=int(a.numel()), flips=int(diff.sum().item()),
                sign_of_zero_corner_flips=int((diff & zero).sum().item()),
                flips_outside_exact_zero=int((diff & ~zero).sum().item()),
                value_equal=bool(torch.equal(a.detach(), b.detach())))


# ------------------------------------------------------------------ gate A


def gate_a_born_as_gdn2(receipt: dict, seed: int) -> dict:
    """The hybrid against a pure-GDN2 twin of the SAME build: one real
    production microbatch; the full-model loss and every slot layer's
    combined output asserted bit-identical (flips counted bitwise)."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import convert_gdn2_to_hybrid

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    pure, _cfg, _summary = matched_arms.build_gdn2(seed)
    hybrid, _cfg2, _summary2 = matched_arms.build_gdn2(seed)
    hybrid = convert_gdn2_to_hybrid(hybrid)
    rows = next(iter(stream.batches("train")))
    x, y, _valid = next(iter(grouped(rows, _cfg["microbatch"])))

    captured = {"pure": {}, "hybrid": {}}
    handles = []

    def hook(side, index):
        def register(_module, _args, output):
            captured[side][index] = output[0].detach().clone()
        return register

    for index in SLOT_LAYERS:
        handles.append(pure.model.blocks[index].layer.gdn2
                       .register_forward_hook(hook("pure", index)))
        handles.append(hybrid.model.blocks[index].layer.gdn2
                       .register_forward_hook(hook("hybrid", index)))
    with torch.no_grad():
        loss_pure = pure.loss(x, y)
        loss_hybrid = hybrid.loss(x, y)
    for handle in handles:
        handle.remove()
    gate = dict(name="A (born as GDN2, bit for bit)")
    gate["loss_bit_identical"] = bool(loss_pure == loss_hybrid)
    gate["loss_pure"] = float(loss_pure)
    gate["loss_hybrid"] = float(loss_hybrid)
    per_layer = {}
    for index in SLOT_LAYERS:
        per_layer[str(index)] = _flip_rows(captured["pure"][index],
                                          captured["hybrid"][index])
    gate["per_slot_layer"] = per_layer
    gate["all_value_equal"] = (all(row["value_equal"]
                                    for row in per_layer.values())
                               if per_layer else None)
    gate["flips_outside_exact_zero"] = sum(
        row.get("flips_outside_exact_zero", 0)
        for row in per_layer.values())
    gate["sign_of_zero_corner_flips"] = sum(
        row.get("sign_of_zero_corner_flips", 0)
        for row in per_layer.values())
    gate["lanes"] = sum(row.get("lanes", 0) for row in per_layer.values())
    gate["note"] = ("the wrapper's output vs the committed layer's own on "
                    "the production microbatch: the slot contribution is "
                    "exactly zero at birth, so the layer IS the committed "
                    "GDN2 layer bit for bit; the only legal escape is the "
                    "IEEE sign-of-zero corner (counted separately)")
    receipt["gate_a"] = gate
    del pure, hybrid, captured
    torch.cuda.empty_cache()
    return gate


def gate_a2_dtype_discipline(receipt: dict) -> dict:
    """Under the production autocast regime the wrapper's output dtype is
    the dense layer's own (the contribution cast to the dense output's
    carrier dtype): the residual stream's dtype is the committed
    machine's own at birth and after.  The invariant is RELATIVE (the
    wrapper must not change whatever dtype the dense side emits): a bf16
    input under the production autocast regime keeps the whole seam in
    bf16, and without autocast the wrapper preserves the input's dtype."""
    class _Dense(nn.Module):  # a stand-in dense layer: passes its input through
        def forward(self, x, h0=None, **kwargs):
            return x.clone(), None

    dense = _Dense()
    mixer = build_slot_mixer(width=64).cuda()
    wrapper = HybridSlotGDN2Layer(dense, mixer, 0).cuda()
    x32 = torch.randn(2, 32, 64, device="cuda")
    xb16 = x32.to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense_out, _ = dense(xb16)
        out, _cache = wrapper(xb16)
    gate = dict(name="A2 (the seam's dtype discipline)")
    gate["contribution_dtype_cast"] = "self.read_gate.to(out.dtype)"
    gate["dense_output_dtype"] = str(dense_out.dtype)
    gate["wrapper_output_dtype"] = str(out.dtype)
    gate["held_under_autocast"] = bool(
        out.dtype == dense_out.dtype == torch.bfloat16)
    with torch.no_grad():
        out32, _ = wrapper(x32)
    gate["held_no_autocast"] = bool(out32.dtype == x32.dtype)
    gate["held_all"] = bool(gate["held_under_autocast"]
                             and gate["held_no_autocast"])
    gate["note"] = ("the wrapper must not change the dense side's carrier "
                    "dtype (the production regime's bf16 residual stream "
                    "stays bf16 through the seam; the real-model receipt is "
                    "gate A's bitwise loss identity)")
    receipt["gate_a2"] = gate
    return gate


# ------------------------------------------------------------------ gate B


def _seam_outputs(mixer, scan_fn, x, dense_out, gate_value):
    """The seam's own expression, walked by ``scan_fn`` (the oracle or the
    committed kernel): the slot mixer's own projections and gates, the
    conv on the payload, the scan, the r_vec read gate, the output
    projection -- then ``dense_out + gate * slot_out``.  Mirrors
    ``ShortConvComposedMixer.mix`` and the wrapper's forward line for
    line (the committed expressions, not new ones)."""
    batch, tokens, _ = x.shape
    context = causal_context(x, mixer.ctx).reshape(
        batch, tokens, mixer.ctx * mixer.width)
    payload = short_conv1d_causal(
        mixer.value(x), mixer.v_conv.weight).view(
        batch, tokens, mixer.heads, mixer.value_dim)
    write_keys = mixer.key(context).view(batch, tokens, mixer.heads,
                                         mixer.key_dim)
    query_keys = mixer.query_key(context).view(batch, tokens, mixer.heads,
                                               mixer.key_dim)
    lam_vec, _lam_inc = mixer.channel_decay(x)
    gates = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
        * torch.sigmoid(mixer.b_proj(x)).view(
            batch, tokens, mixer.heads, mixer.value_dim)
    result = scan_fn(payload, write_keys, query_keys, lam_vec, gates)
    reads = result["reads"] if isinstance(result, dict) else result.reads
    r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
        * torch.sigmoid(mixer.g_proj(x)).view(
            batch, tokens, mixer.heads, mixer.value_dim)
    reads = reads * r_vec
    slot_out = mixer.out(reads.reshape(batch, tokens,
                                       mixer.heads * mixer.value_dim))
    return dense_out + gate_value * slot_out, slot_out


def gate_b_seam_gradcheck(receipt: dict) -> dict:
    """MANDATORY: the fp64 gradcheck THROUGH THE HYBRID SEAM (the
    slot-in-dense-stack boundary), both gated combinations, over the dense
    leaf, the gate, the input and every slot parameter; then the kernel
    path's seam gradients within the measured fp32-vs-fp64 floors."""
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer

    gate = dict(name="B (the fp64 gradcheck through the hybrid seam)")
    policy = slot_policy()
    batch, tokens, width = 2, 12, 48
    value_dim, key_dim, rank = 8, 4, 5

    # (i) the fp64 gradcheck through the seam, both gated combinations
    gradchecks = {}
    for gated in (True, False):
        state = torch.random.get_rng_state()
        torch.manual_seed(29 if gated else 31)
        mixer = ShortConvComposedMixer(
            width=width, heads=1, value_dim=value_dim, rank=rank,
            rank_max=rank, key_dim=key_dim, policy=policy, ctx=4,
            segment_tokens=64, birth="identity").cuda().double()
        torch.random.set_rng_state(state)
        dense = torch.randn(batch, tokens, width, dtype=torch.float64,
                            device="cuda").requires_grad_(True)
        gate_value = torch.zeros((), dtype=torch.float64,
                                 device="cuda").requires_grad_(True)
        x = torch.randn(batch, tokens, width, dtype=torch.float64,
                        device="cuda").requires_grad_(True)

        def scan_fn(payload, write_keys, query_keys, lam_vec, gates_,
                    gated=gated):
            return composed_oracle_scan(
                payload, write_keys, query_keys, lam_vec, gates_,
                policy=policy, gated=gated, rank=rank, rank_max=rank)

        def f(dense_, gate_, x_, *params):
            out, _slot = _seam_outputs(mixer, scan_fn, x_, dense_, gate_)
            return out

        inputs = [dense, gate_value, x] + list(mixer.parameters())
        gradchecks[str(gated)] = bool(torch.autograd.gradcheck(
            f, tuple(inputs), eps=1e-6, atol=1e-4, rtol=1e-4,
            nondet_tol=0.0))
    gate["fp64_gradcheck_through_the_seam"] = gradchecks

    # (ii) the kernel path's seam gradients vs the measured floors, at a
    # production-geometry fixture (the committed composed_scan op pair)
    b, t, w = 2, 256, 64
    dense = torch.randn(b, t, w,
                        generator=torch.Generator().manual_seed(51)).cuda()
    x = torch.randn(b, t, w,
                    generator=torch.Generator().manual_seed(52)).cuda()
    gate_value = torch.tensor(0.3)
    upstream = torch.randn(b, t, w,
                           generator=torch.Generator().manual_seed(1234)
                           ).cuda()

    def chain(scan_fn, dtype):
        dense_ = dense.clone().to(dtype).requires_grad_(True)
        # a FRESH leaf per chain (a shared `.to()` fixture would go non-leaf
        # on the second chain: `.to(float32)` returns the same tensor, so
        # `requires_grad_` marks the fixture itself)
        gate_ = torch.tensor(0.3, device=dense_.device,
                             dtype=dtype).requires_grad_(True)
        x_ = x.clone().to(dtype).requires_grad_(True)
        state = torch.random.get_rng_state()
        torch.manual_seed(37)
        mixer_ = build_slot_mixer(width=w).cuda().to(dtype)
        torch.random.set_rng_state(state)
        out, _slot = _seam_outputs(mixer_, scan_fn, x_, dense_, gate_)
        (out * upstream.to(dtype)).sum().backward()
        return dict(ddense=dense_.grad, dgate=gate_.grad.clone(),
                    dx=x_.grad)

    rows = {}
    kern = chain(lambda *a: composed_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128,
        segment_tokens=64), torch.float32)
    orac32 = chain(lambda *a: composed_oracle_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128),
        torch.float32)
    orac64 = chain(lambda *a: composed_oracle_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128),
        torch.float64)
    for name in ("ddense", "dgate", "dx"):
        floor = (orac32[name].float()
                 - orac64[name].float()).abs().max().item()
        deviation = (kern[name].float()
                     - orac32[name].float()).abs().max().item()
        rows[name] = dict(floor=floor, deviation=deviation,
                          within_noise=bool(
            deviation <= NOISE_FACTOR * max(floor, 1e-6)
            or deviation <= 1e-5))
    gate["kernel_seam_gradients_vs_floors"] = rows
    receipt["gate_b"] = gate
    return gate


# ------------------------------------------------------------------ gate C


def gate_c_open_signal(receipt: dict) -> dict:
    """The open signal and the inert birth at the zero-born gate, on the
    seam's own fixture: dL/dgate nonzero (the open signal), every other
    slot parameter's gradient exactly zero, and the store fills (the ops
    fire) -- the machinery runs, nothing reads out."""
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        OP_NAMES)
    b, t, w = 2, 256, 64
    state = torch.random.get_rng_state()
    torch.manual_seed(41)
    mixer = build_slot_mixer(width=w).cuda()
    torch.random.set_rng_state(state)
    dense = torch.randn(b, t, w, device="cuda")
    x = torch.randn(b, t, w, device="cuda")
    upstream = torch.randn(b, t, w,
                           generator=torch.Generator().manual_seed(1234)
                           ).cuda()
    gate_value = torch.zeros((), device="cuda").requires_grad_(True)
    out, _slot = _seam_outputs(mixer, lambda *a: composed_scan(
        *a, policy=slot_policy(), gated=True, rank=128, rank_max=128,
        segment_tokens=64), x, dense, gate_value)
    (out * upstream).sum().backward()
    slot_grads = {name: param.grad for name, param
                 in mixer.named_parameters() if param.grad is not None}
    inert = bool(all(bool((g == 0).all()) for g in slot_grads.values()))
    max_inert = float(max((g.abs().max().item() for g in slot_grads.values()),
                          default=0.0))
    # the store fills: the ops fire on the same fixture (the oracle's own
    # recorded schedule)
    with torch.no_grad():
        ref = composed_oracle_scan(
            short_conv1d_causal(
                mixer.value(x), mixer.v_conv.weight).view(
                b, t, mixer.heads, mixer.value_dim),
            mixer.key(causal_context(x, mixer.ctx).reshape(
                b, t, mixer.ctx * w)).view(b, t, mixer.heads,
                                           mixer.key_dim),
            mixer.query_key(causal_context(x, mixer.ctx).reshape(
                b, t, mixer.ctx * w)).view(b, t, mixer.heads,
                                       mixer.key_dim),
            mixer.channel_decay(x)[0],
            (1.0 - mixer.write_scalar.view(1, 1, -1, 1)
             * torch.sigmoid(mixer.b_proj(x)).view(
                 b, t, mixer.heads, mixer.value_dim)),
            policy=slot_policy(), gated=True, rank=128, rank_max=128)
    hist = op_histogram(ref["schedule"])
    gate = dict(name="C (the open signal and the inert birth)")
    gate["gate_born_at_exact_zero"] = bool(gate_value.item() == 0.0)
    gate["gate_gradient"] = float(gate_value.grad.item())
    gate["open_signal_nonzero"] = bool(gate_value.grad.item() != 0.0)
    gate["slot_parameters_all_exactly_zero_grad"] = inert
    gate["slot_max_abs_grad_at_birth"] = max_inert
    gate["store_fills"] = dict(
        op_histogram={k: int(v) for k, v in hist.items()},
        append_or_correct_fires=bool(hist.get("append", 0)
                                     + hist.get("correct", 0) > 0))
    receipt["gate_c"] = gate
    return gate


# ------------------------------------------------------------------ gate D


def gate_d_wall(receipt: dict) -> dict:
    """The honest wall: the slot branch's own forward cost per (slot
    layer, window-group) at the registered production shape (batch 8 x
    2048 tokens at the host width 640), stated against the K20/K21
    receipts' own measured composed-mixer cost."""
    b, t, w = 8, 2048, 640
    state = torch.random.get_rng_state()
    torch.manual_seed(43)
    mixer = build_slot_mixer(width=w).cuda()
    torch.random.set_rng_state(state)
    x = torch.randn(b, t, w, device="cuda")

    def forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mixer(x)

    for _ in range(5):
        forward()
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        torch.cuda.synchronize()
        started = time.monotonic()
        forward()
        torch.cuda.synchronize()
        samples.append((time.monotonic() - started) * 1000.0)
    ordered = sorted(samples)
    gate = dict(name="D (the slot branch's honest wall)")
    gate["median_ms_per_window_group"] = round(ordered[len(ordered) // 2], 3)
    gate["min_ms"] = round(min(ordered), 3)
    gate["max_ms"] = round(max(ordered), 3)
    gate["shape"] = dict(batch=b, tokens=t, width=w)
    gate["cited"] = ("the K20 receipt: the committed composed mixer at "
                     "width 576 costs 23.40 ms fwd per (layer, "
                     "window-group); the K21 profile: 93.1% one Triton "
                     "kernel; the hybrid pays this on 3 of 24 layers, "
                     "TWICE per training step (the checkpointed block "
                     "re-runs the forward in backward)")
    receipt["gate_d"] = gate
    return gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "receipts" / "k22_gates.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    receipt = dict(
        probe="k22_gate_receipt", seed=args.seed,
        note="the K22 hybrid's birth receipts at the seam: born as GDN2 "
             "bit for bit, the dtype discipline, the fp64 gradcheck "
             "through the slot-in-dense-stack boundary, the open signal "
             "and the inert birth, the honest wall")
    gate_a_born_as_gdn2(receipt, args.seed)
    gate_a2_dtype_discipline(receipt)
    gate_b_seam_gradcheck(receipt)
    gate_c_open_signal(receipt)
    gate_d_wall(receipt)
    ok = (receipt["gate_a"]["loss_bit_identical"]
          and receipt["gate_a"]["flips_outside_exact_zero"] == 0
          and receipt["gate_a2"]["held_all"]
          and all(receipt["gate_b"]["fp64_gradcheck_through_the_seam"]
                  .values())
          and all(row["within_noise"] for row in
                  receipt["gate_b"]["kernel_seam_gradients_vs_floors"]
                  .values())
          and receipt["gate_c"]["open_signal_nonzero"]
          and receipt["gate_c"]["slot_parameters_all_exactly_zero_grad"]
          and receipt["gate_c"]["store_fills"]["append_or_correct_fires"])
    receipt["all_checks"] = bool(ok)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in
                      ("probe", "all_checks")}, indent=1), flush=True)
    print(json.dumps(receipt["gate_a"]["per_slot_layer"], indent=1),
          flush=True)
    if not ok:
        raise SystemExit("the K22 gate receipts FAILED -- escalate, not "
                         "excuse")


if __name__ == "__main__":
    main()
