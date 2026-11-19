"""K11B -- the frozen-gates falsifier failure diagnostic (a receipt for the
escalation; NOT a probe, NOT a patch: the engine is committed and read-only).

The falsifier control FAILED its step-1 bar: the frozen p1 arm's step-1 train
loss 5.785136661889908 != the banked same-seed row's 5.785070835425993
(delta ~6.6e-5).  Step 1 is forward-only (before any optimizer step), so the
added gate parameters are provably inert there; a divergence means the
k11b-converted model's FORWARD at birth is not bit-identical to the banked
K10 winner's -- an engine-level parity break at the production regime, NOT
the gates learning.

This script localizes it, honestly, in three receipts:

  R1  the scan-level birth parity at the PRODUCTION regime (bf16 payload and
      keys, an FP32 decay vector -- gate A's fixture used bf16 decay -- and
      the (1,1,1,1) dummy gates tensor the p1 arm passes): K10's own launch
      vs the k11b launch, bitwise, at the registered shape and the tuned
      winner's own policy constants.
  R2  the model-level step-1 reproduction: build the banked winner (seed
      3401), compute the probe's own step-1 train loss on the real flagship
      stream's first 64-window batch -- first UNCONVERTED (vs the banked
      row), then the SAME model converted to arm p1 at birth (vs the
      falsifier's row).
  R3  the layer-by-layer localization on the first microbatch: per-block
      outputs of the unconverted model vs the converted one, the first
      differing layer named, then that layer's mixer dissected (payload /
      keys / lam_vec-vs-decay / the two scans' reads).

Run leased (cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=... .venv/bin/python \
        k11b_runs/k11b_birth_diagnostic.py --out k11b_runs/birth_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
    import _round_f32
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
    import (_k10_forward_launch, _seq_warps, separated_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
    import (_k11b_forward_launch, channel_gated_scan,
            convert_model_to_channel_gates)
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import MICROBATCH, TUNED_WINNER
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)
from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
    import _build_k10
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
    grouped)

BANKED_STEP1 = 5.785070835425993
FALSIFIER_STEP1 = 5.785136661889908
WINDOWS_PER_STEP = 64


def prod_policy() -> RoutingPolicy:
    return RoutingPolicy(ops=True,
                         tau_match=TUNED_WINNER["tau_match"],
                         tau_merge=TUNED_WINNER["tau_merge"],
                         tau_pop=TUNED_WINNER["tau_pop"],
                         read_scale=TUNED_WINNER["read_scale"])


def pol32_of(policy: RoutingPolicy):
    return (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
            _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
            _round_f32(policy.read_scale))


def receipt_r1(receipt: dict) -> dict:
    """Scan-level birth parity at the production regime."""
    torch.manual_seed(1234)
    b, T, h, vp, rank, kd = 8, 2048, 2, 128, 128, 16
    payload = torch.randn(b, T, h, vp, dtype=torch.bfloat16, device="cuda")
    wk = torch.randn(b, T, h, kd, dtype=torch.bfloat16, device="cuda")
    qk = torch.randn(b, T, h, kd, dtype=torch.bfloat16, device="cuda")
    lam = (torch.rand(b, T, h, dtype=torch.float32, device="cuda")
           * 0.15 + 0.85).clamp(min=1e-4, max=1.0 - 1e-6)
    lam_vec = lam.unsqueeze(-1).expand(b, T, h, vp).contiguous()
    dummy = payload.new_ones((1, 1, 1, 1))
    state = torch.zeros(b, h, rank, vp, dtype=torch.float32, device="cuda")
    live = torch.zeros(b, h, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(b, h, rank, kd, dtype=torch.float32, device="cuda")
    policy = prod_policy()
    pol32 = pol32_of(policy)
    warps = _seq_warps(rank, vp)
    k10 = _k10_forward_launch(payload, wk, qk, lam, state, live, keys,
                              pol32, rank, 64, collect=False, warps=warps,
                              stages=3)
    k11b = _k11b_forward_launch(payload, wk, qk, lam_vec, dummy, state, live,
                                keys, pol32, rank, 64, False, collect=False,
                                warps=warps, stages=3)
    rows = {}
    for name in ("reads", "state", "live", "keys"):
        a, c = k10[name], k11b[name]
        rows[name] = dict(bit_identical=bool(torch.equal(a, c)),
                          differing=(int((a != c).sum())
                                     if a.shape == c.shape else None),
                          max_abs_delta=float((a.float() - c.float())
                                              .abs().max())
                          if a.shape == c.shape and a.is_floating_point()
                          else None)
    receipt["r1_scan_level_production_regime"] = dict(
        fixture=dict(batch=b, tokens=T, heads=h, rank=rank, value_dim=vp,
                     key_dim=kd, segment_tokens=64,
                     payload_dtype="bf16 (autocast)", keys_dtype="bf16",
                     decay_dtype="fp32 (production; gate A used bf16)",
                     gates="the (1,1,1,1) bf16 dummy the p1 arm passes",
                     warps=warps, policy=policy.record()),
        channels=rows,
        verdict=("bit-identical at the production scan regime"
                 if all(r["bit_identical"] for r in rows.values())
                 else "DIVERGES at the production scan regime"))
    return receipt["r1_scan_level_production_regime"]


def _first_step_rows(stream, microbatch: int):
    batches = iter(stream.batches("train"))
    batch_rows = next(batches)
    n = sum(len(r[3]) for r in batch_rows)
    chunks = list(grouped(batch_rows, microbatch))
    return n, chunks


def _step1_loss(model, chunks, n: int) -> float:
    loss_sum = 0.0
    per_micro = []
    for x, y, valid in chunks:
        loss = model.loss(x, y)
        loss_sum += float(loss.detach()) * valid / n
        per_micro.append(float(loss.detach()))
    return loss_sum, per_micro


def receipt_r2(model, chunks, n: int, receipt: dict) -> dict:
    unconverted, per_micro_k10 = _step1_loss(model, chunks, n)
    model = convert_model_to_channel_gates(model, arm="p1")
    model = model.cuda().train()
    converted, per_micro_k11b = _step1_loss(model, chunks, n)
    receipt["r2_model_level_step1"] = dict(
        n_targets=n, microbatches=len(chunks),
        unconverted_step1_loss=unconverted,
        unconverted_vs_banked=dict(
            banked=BANKED_STEP1, delta=unconverted - BANKED_STEP1,
            bit_identical=unconverted == BANKED_STEP1),
        converted_step1_loss=converted,
        converted_vs_falsifier=dict(
            falsifier=FALSIFIER_STEP1, delta=converted - FALSIFIER_STEP1,
            bit_identical=converted == FALSIFIER_STEP1),
        per_microbatch_k10=per_micro_k10,
        per_microbatch_k11b=per_micro_k11b,
        verdict=("the unconverted build reproduces the banked row and the "
                 "converted one reproduces the falsifier's -- the break is "
                 "IN the k11b conversion's forward at birth"
                 if unconverted == BANKED_STEP1
                    and converted == FALSIFIER_STEP1 else
                 "unexpected: one of the two rows did not reproduce -- "
                 "see the numbers"))
    receipt["_converted_model"] = model
    return receipt["r2_model_level_step1"]


def receipt_r3(model, chunks, receipt: dict) -> dict:
    """Layer-by-layer localization on microbatch 0, then the mixer
    dissection at the first differing layer.  (Built on a FRESH model this
    time: r2 converted its model in place, so the first version of this
    receipt compared the converted model against itself -- invalid, redone.)"""
    x, y, _valid = chunks[0]
    outs = {}

    def capture(tag):
        handles = []
        table = outs.setdefault(tag, [])

        def hook(mod, args, output):
            table.append(output.detach().clone())
        for block in model.blocks:
            handles.append(block.register_forward_hook(hook))
        with torch.no_grad():
            model.loss(x, y)
        for h in handles:
            h.remove()

    # the mixer input of layer 0, captured BEFORE the conversion
    mixer_inputs = {}

    def pre_hook(mod, args):
        mixer_inputs["x0"] = args[0].detach().clone()
    model.blocks[0].mixer.register_forward_pre_hook(pre_hook)
    capture("k10")
    for handle in list(getattr(model.blocks[0].mixer,
                               "_forward_pre_hooks", {}).values()):
        pass
    capture_k10_mixer = dict(mixer_inputs)
    model = convert_model_to_channel_gates(model, arm="p1").cuda().train()
    capture("k11b")
    first_diff = None
    per_layer = {}
    for i, (a, c) in enumerate(zip(outs["k10"], outs["k11b"])):
        same = bool(torch.equal(a, c))
        per_layer[str(i)] = dict(
            bit_identical=same,
            differing=int((a != c).sum()) if not same else 0,
            max_abs_delta=float((a.float() - c.float()).abs().max())
            if not same else 0.0)
        if not same and first_diff is None:
            first_diff = i
    receipt["r3_layer_localization"] = dict(
        note="per-BLOCK outputs (post-block, hidden states) on microbatch 0, "
             "the k10 pass captured BEFORE the in-place conversion",
        first_differing_layer=first_diff,
        per_layer=per_layer)

    # ---- r4: the production-data scan comparison -- the REAL layer-0
    # tensors (captured above, pre-conversion) fed to BOTH kernels
    x0 = capture_k10_mixer["x0"]
    mixer = model.blocks[0].mixer          # the converted one; shared modules
    batch, tokens, _ = x0.shape
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import causal_context
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        context = causal_context(x0, mixer.ctx).reshape(
            batch, tokens, mixer.ctx * x0.shape[-1])
        payload = mixer.value(x0).view(batch, tokens, mixer.heads,
                                       mixer.value_dim)
        write_keys = mixer.key(context).view(batch, tokens, mixer.heads,
                                            mixer.key_dim)
        query_keys = mixer.query_key(context).view(
            batch, tokens, mixer.heads, mixer.key_dim)
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        DECAY_FLOOR)
    # BOTH decay forms UNDER AUTOCAST (the production regime; the first
    # version computed them outside autocast -- a different, non-production
    # value regime)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        decay = torch.sigmoid(mixer.decay(x0)).clamp(min=DECAY_FLOOR,
                                                      max=1.0 - 1e-6)
        lam_vec, _lam_inc = mixer.channel_decay(x0)
    r4 = dict(
        note="the REAL production layer-0 tensors (microbatch 0, captured "
             "pre-conversion) fed to BOTH kernels with BOTH decay forms "
             "computed UNDER AUTOCAST (the production regime); plus the "
             "birth lam_vec-vs-decay value check on the same data",
        x0_dtype=str(x0.dtype),
        lam_vec_vs_decay=dict(
            values_equal=bool(torch.equal(
                lam_vec, decay.unsqueeze(-1).expand_as(lam_vec)
                .to(lam_vec.dtype))),
            lam_vec_dtype=str(lam_vec.dtype), decay_dtype=str(decay.dtype)))
    lam_vec_c = lam_vec.contiguous()
    r4["lam_vec_vs_decay"]["values_equal_note"] = (
        "lam_vec is the autocast channel_decay output; decay is the K10 "
        "mix's own autocast expression -- the two production carriers")
    state = torch.zeros(batch, mixer.heads, mixer.rank, mixer.value_dim,
                         dtype=torch.float32, device="cuda")
    live = torch.zeros(batch, mixer.heads, mixer.rank, dtype=torch.bool,
                       device="cuda")
    keys = torch.zeros(batch, mixer.heads, mixer.rank, mixer.key_dim,
                       dtype=torch.float32, device="cuda")
    policy = mixer.policy
    pol32 = pol32_of(policy)
    warps = _seq_warps(mixer.rank, mixer.value_dim)
    k10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                              state, live, keys, pol32, mixer.rank,
                              mixer.segment_tokens, collect=True, warps=warps,
                              stages=3)
    dummy = payload.new_ones((1, 1, 1, 1))
    k11b = _k11b_forward_launch(payload, write_keys, query_keys, lam_vec_c,
                                dummy, state, live, keys, pol32,
                                mixer.rank, mixer.segment_tokens, False,
                                collect=True, warps=warps, stages=3)
    rows = {}
    for name in ("reads", "state", "live", "keys", "weights"):
        a, c = k10[name], k11b[name]
        rows[name] = dict(
            bit_identical=bool(torch.equal(a, c)),
            differing=int((a != c).sum()) if a.shape == c.shape else None,
            max_abs_delta=float((a.float() - c.float()).abs().max())
            if a.shape == c.shape and a.is_floating_point() else None)
    sched_flips = {}
    for name in ("slot", "erase", "op", "live_after"):
        a, c = k10[name], k11b[name]
        sched_flips[name] = int((a != c).sum())
    rows["schedule"] = dict(flips=sched_flips,
                            any_flip=any(v != 0 for v in
                                        sched_flips.values()))
    r4["channels"] = rows
    # localize the first differing read element
    if not rows["reads"]["bit_identical"]:
        diff = (k10["reads"] != k11b["reads"])
        idx = diff.nonzero()
        first = idx[0].tolist()
        r4["first_differing_read"] = dict(
            index=first,
            count=int(idx.shape[0]),
            k10_value=float(k10["reads"][tuple(first)]),
            k11b_value=float(k11b["reads"][tuple(first)]),
            k10_schedule_at_token=dict(
                slot=k10["slot"][first[0], first[1], first[2]].tolist(),
                op=k10["op"][first[0], first[1], first[2]].tolist()),
            k11b_schedule_at_token=dict(
                slot=k11b["slot"][first[0], first[1], first[2]].tolist(),
                op=k11b["op"][first[0], first[1], first[2]].tolist()))
    receipt["r4_production_data_scan_comparison"] = r4

    # ---- r5: dissect the first differing layer -- its REAL tensors fed to
    # BOTH kernels, the op histogram, the per-channel lam check, and the
    # first differing read localized to its token and its op
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        op_histogram)
    L = first_diff if first_diff is not None else 0
    layers_to_dissect = sorted(set([0, L, L + 1]))
    mixer_inputs = {}

    def pre_hook_factory(index):
        def hook(mod, args):
            mixer_inputs[index] = args[0].detach().clone()
        return hook
    handles = [model.blocks[i].mixer.register_forward_pre_hook(
        pre_hook_factory(i)) for i in layers_to_dissect]
    with torch.no_grad():
        model.loss(x, y)
    for h in handles:
        h.remove()
    dissections = {}
    for i in layers_to_dissect:
        x_i = mixer_inputs[i]
        mixer = model.blocks[i].mixer
        batch, tokens, _ = x_i.shape
        # BOTH decay forms computed UNDER AUTOCAST, exactly as each mixer's
        # mix does in production (the earlier r5 called channel_decay
        # outside autocast -- different VALUES, invalid comparison)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            context = causal_context(x_i, mixer.ctx).reshape(
                batch, tokens, mixer.ctx * x_i.shape[-1])
            payload = mixer.value(x_i).view(batch, tokens, mixer.heads,
                                           mixer.value_dim)
            write_keys = mixer.key(context).view(
                batch, tokens, mixer.heads, mixer.key_dim)
            query_keys = mixer.query_key(context).view(
                batch, tokens, mixer.heads, mixer.key_dim)
            decay = torch.sigmoid(mixer.decay(x_i)).clamp(
                min=DECAY_FLOOR, max=1.0 - 1e-6)
            lam_vec, lam_inc = mixer.channel_decay(x_i)
            payload10 = mixer.value(x_i).view(batch, tokens, mixer.heads,
                                              mixer.value_dim)
        row = dict(
            layer=i, x_dtype=str(x_i.dtype), payload_dtype=str(payload.dtype),
            decay_dtype=str(decay.dtype), lam_vec_dtype=str(lam_vec.dtype),
            lam_inc_dtype=str(lam_inc.dtype),
            lam_vec_values_equal_decay=bool(torch.equal(
                lam_vec, decay.unsqueeze(-1).expand_as(lam_vec)
                .to(lam_vec.dtype))),
            lam_vec_max_value_delta=float((lam_vec
                - decay.unsqueeze(-1).expand_as(lam_vec)
                .to(lam_vec.dtype)).abs().max()))
        state = torch.zeros(batch, mixer.heads, mixer.rank, mixer.value_dim,
                            dtype=torch.float32, device="cuda")
        live = torch.zeros(batch, mixer.heads, mixer.rank, dtype=torch.bool,
                           device="cuda")
        keys = torch.zeros(batch, mixer.heads, mixer.rank, mixer.key_dim,
                           dtype=torch.float32, device="cuda")
        pol32 = pol32_of(mixer.policy)
        warps = _seq_warps(mixer.rank, mixer.value_dim)
        k10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                                  state, live, keys, pol32, mixer.rank,
                                  mixer.segment_tokens, collect=True,
                                  warps=warps, stages=3)
        dummy = payload.new_ones((1, 1, 1, 1))
        k11b = _k11b_forward_launch(payload, write_keys, query_keys,
                                    lam_vec.contiguous(), dummy, state, live,
                                    keys, pol32, mixer.rank,
                                    mixer.segment_tokens, False,
                                    collect=True, warps=warps, stages=3)
        sched10 = dict(slot=k10["slot"], erase=k10["erase"],
                       payload=k10["payload"], op=k10["op"],
                       live_after=k10["live_after"])
        row["op_histogram"] = op_histogram(type("S", (), sched10)())
        channels = {}
        for name in ("reads", "state", "live", "keys", "weights"):
            a, c = k10[name], k11b[name]
            channels[name] = dict(
                bit_identical=bool(torch.equal(a, c)),
                differing=int((a != c).sum()) if a.shape == c.shape else None,
                max_abs_delta=float((a.float() - c.float()).abs().max())
                if a.shape == c.shape and a.is_floating_point() else None)
        flips = {}
        for name in ("slot", "erase", "op", "live_after"):
            flips[name] = int((k10[name] != k11b[name]).sum())
        channels["schedule"] = dict(flips=flips,
                                    any_flip=any(v != 0 for v in
                                                 flips.values()))
        row["channels"] = channels
        if not channels["reads"]["bit_identical"]:
            diff = (k10["reads"] != k11b["reads"])
            idx = diff.nonzero()
            first = idx[0].tolist()
            b_, t_, h_, c_ = first
            row["first_differing_read"] = dict(
                index=first, count=int(idx.shape[0]),
                k10_value=float(k10["reads"][tuple(first)]),
                k11b_value=float(k11b["reads"][tuple(first)]),
                token_ops=dict(
                    k10=k10["op"][b_, t_, h_].tolist(),
                    k11b=k11b["op"][b_, t_, h_].tolist()),
                token_slots=dict(
                    k10=k10["slot"][b_, t_, h_].tolist(),
                    k11b=k11b["slot"][b_, t_, h_].tolist()),
                tokens_with_any_read_diff=int(
                    diff.any(dim=-1).sum()))
        # ---- r6: THE CLAMP-BOUND CARRIER CORNER receipt -- demonstrated
        # directly on this layer's real data AND as pure tensor semantics:
        # the incumbent's BF16 carrier rounds the clamp bound 1-1e-6 to
        # 1.0, while channel_decay's factor multiply promotes to FP32,
        # whose clamp preserves fp32(1-1e-6) -- every position whose
        # bf16 sigmoid output is exactly 1.0 diverges at birth by exactly
        # 1.0 - fp32(1-1e-6)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits_ac = mixer.decay(x_i)
            lam_inc_ac = torch.sigmoid(logits_ac)
        lam_pos = (lam_vec != decay.unsqueeze(-1).expand_as(lam_vec)
                   .to(lam_vec.dtype))
        differing_lam = lam_pos.any(dim=-1)
        row["clamp_bound_carrier_corner"] = dict(
            note="lam_inc positions where the bf16 sigmoid output is "
                 "exactly 1.0 clamp to 1.0 in the incumbent's BF16 carrier "
                 "(bf16(1-1e-6) == 1.0) but to fp32(1-1e-6) = 0.999999 in "
                 "channel_decay's FP32-promoted carrier -- the birth forms "
                 "differ by ~1.013e-6 at exactly those positions",
            logits_dtype=str(logits_ac.dtype),
            differing_lam_positions=int(differing_lam.sum()),
            total_positions=int(differing_lam.numel()),
            differing_lam_inc_values=[
                float(v) for v in lam_inc_ac[differing_lam].unique()[:8]],
            differing_decay_values=[
                float(v) for v in decay[differing_lam].unique()[:8]],
            differing_lam_vec_values=[
                float(v) for v in lam_vec[lam_pos].unique()[:8]],
            expected_delta=float(
                torch.tensor(1.0, dtype=torch.float32)
                - torch.tensor(1.0 - 1e-6, dtype=torch.float32)),
            pure_tensor_semantics=dict(
                bf16_clamp_of_one=float(
                    torch.tensor(1.0, dtype=torch.bfloat16)
                    .clamp(max=1.0 - 1e-6)),
                fp32_clamp_of_one=float(
                    torch.tensor(1.0, dtype=torch.float32)
                    .clamp(max=1.0 - 1e-6)),
                bf16_bound_in_fp32=float(
                    torch.tensor(1.0 - 1e-6, dtype=torch.bfloat16)
                    .float())))
        dissections[str(i)] = row
    receipt["r5_first_diff_layer_dissection"] = dict(
        first_differing_layer=first_diff,
        layers=dissections)
    return receipt["r3_layer_localization"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="k11b_runs/birth_diagnostic.json")
    parser.add_argument("--skip-r3", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    L.check_python()
    receipt = dict(probe="k11b_birth_diagnostic",
                   banked_step1=BANKED_STEP1, falsifier_step1=FALSIFIER_STEP1)
    receipt_r1(receipt)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import causal_context  # noqa: F401 (used by r3/r4/r5)
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    model, optimizer_config, _summary = _build_k10(dict(TUNED_WINNER), 3401)
    microbatch = optimizer_config.get("microbatch", MICROBATCH)
    n, chunks = _first_step_rows(stream, microbatch)
    receipt_r2(model, chunks, n, receipt)
    receipt.pop("_converted_model", None)
    if not args.skip_r3:
        # r2 converted its model in place; r3 needs the UNCONVERTED forward,
        # so rebuild fresh (same seed, same build order)
        del model
        torch.cuda.empty_cache()
        model, _cfg, _s = _build_k10(dict(TUNED_WINNER), 3401)
        receipt_r3(model, chunks, receipt)
    Path(args.out).write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k in ("r1_scan_level_production_regime",
                               "r2_model_level_step1",
                               "r3_layer_localization")}, indent=1)[:8000])


if __name__ == "__main__":
    main()
