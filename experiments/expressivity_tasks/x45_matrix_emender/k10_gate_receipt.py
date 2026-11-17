"""K10 -- the gate receipts: gates A, B and D at the registered shape.

Writes ``receipts/k10_gates.json`` (the K6 ``k6_gate_receipt.py`` pattern):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs): the
  schedule fields, the payload lane, the read WEIGHTS, the pre-decision
  liveness, the exit state/liveness AND the exit key table bit-equal against
  the eager per-token oracle; the read at the measured fp32-vs-fp64 floor; the
  KEY-COLLAPSE watch (``span_key_std``) on the same fixture.
* **Gate B** -- the fp64 gradcheck of the oracle (every adjoint branch), the
  fp32-vs-fp64 floor per output measured on the gate fixture, the kernel
  path's six gradients vs the floor (``NOISE_FACTOR`` 8x), and the
  segment-recompute bit-equality on both channels (value via K4's replay, key
  via the key-lane replay).
* **Gate D** -- block-call walls (forward/backward per (layer, window) scan at
  the registered shape), the register/spill count of the forward kernel, and
  the honest key-channel cost statement against K6's banked walls.

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k10_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k10_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_backward_body, _k10_forward_launch, separated_available,
    separated_sequential_oracle_scan, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K6 gate receipt's own fixture, the key channel
# added at the registration's key_dim 16).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64


def registered_fixture(seed: int = 77, batch: int = BATCH, tokens: int = TOKENS,
                       dtype: torch.dtype = torch.bfloat16):
    """The K5A-bar fixture: payload AND write-key directions from a hot set of
    ``distinct`` fixed directions (the identity-cache regime).  Gate A runs it
    bf16 (the registered autocast regime); gate B runs it fp32 (the floor
    instrument's regime, K6's own choice -- a bf16 leaf would quantise the
    oracle's gradients identically in fp32 and fp64 and make the floor 0)."""
    generator = torch.Generator().manual_seed(seed)
    distinct = 512
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=generator)
    key_dirs = torch.randn(distinct, KEY_DIM, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, HEADS), generator=generator)
    payload = payload_dirs[choice].to(dtype=dtype).cuda()
    write_keys = key_dirs[choice].to(dtype=dtype).cuda()
    query_keys = (torch.randn(batch, tokens, HEADS, KEY_DIM, generator=generator)
                  .to(dtype=dtype).cuda())
    decay = torch.full((batch, tokens, HEADS), 0.95).to(dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay


def gate_a(receipt: dict) -> dict:
    payload, write_keys, query_keys, decay = registered_fixture()
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(BATCH, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    started = time.monotonic()
    out = _k10_forward_launch(payload, write_keys, query_keys, decay, state,
                              live, keys, POL32, RANK, SEGMENT_TOKENS,
                              collect=True, warps=8, stages=3)
    torch.cuda.synchronize()
    kernel_wall = time.monotonic() - started
    started = time.monotonic()
    ref = separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                           decay, policy=POLICY, rank=RANK,
                                           rank_max=RANK, collect=True)
    oracle_wall = time.monotonic() - started
    with torch.no_grad():
        ref64 = separated_sequential_oracle_scan(
            payload.double(), write_keys.double(), query_keys.double(),
            decay.double(), policy=POLICY, rank=RANK, rank_max=RANK)
    rows = {}
    flips = 0
    lanes = 0
    for name in ("slot", "erase", "op", "live_after"):
        a, b = getattr(ref["schedule"], name), out[name]
        equal = bool(torch.equal(a, b))
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[f"schedule_{name}"] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                                         bit_equal=equal)
    for name in ("weights", "live_pre", "state", "live", "keys"):
        a, b = ref[name], out[name]
        equal = bool(torch.equal(a, b))
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[name] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                          bit_equal=equal)
    a, b = ref["schedule"].payload, out["payload"]
    equal = bool(torch.equal(a, b))
    flips += int((a != b).sum())
    lanes += a.numel()
    rows["payload"] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                           bit_equal=equal)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    read_scale = ref64["reads"].float().abs().max().item()
    hist = op_histogram(ref["schedule"])
    spread = span_key_std(write_keys.float())
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, dtype="bf16 inputs, "
                     "fp32 defined-order math", segment_tokens=SEGMENT_TOKENS,
                     distinct_directions=512),
        channels=rows,
        total=dict(lanes=lanes, flips=flips),
        read=dict(floor_fp32_vs_fp64=floor, deviation=deviation,
                  ratio_vs_floor=deviation / max(floor, 1e-30),
                  read_scale=read_scale,
                  within_noise_factor=bool(deviation
                                           <= NOISE_FACTOR * max(floor, 1e-30))),
        op_histogram=hist,
        key_collapse_watch=dict(
            span_key_std_min=float(spread.min()),
            span_key_std_mean=float(spread.mean()),
            guard_floor=0.05,
            within_guard=bool((spread >= 0.05).all())),
        walls=dict(kernel_forward_s=round(kernel_wall, 3),
                   eager_oracle_s=round(oracle_wall, 1)),
    )
    # the register report from the compiled kernel's own metadata
    try:
        meta = out["kernel"]
        gate["forward_kernel_resources"] = dict(
            num_warps=8,
            registers=getattr(meta, "n_regs", None),
            spills=getattr(meta, "n_spills", None))
    except Exception:
        pass
    receipt["gate_a"] = gate
    return gate


def gate_b(receipt: dict) -> dict:
    # 1. the fp64 gradcheck of the oracle (the instrument validates first).
    generator = torch.Generator().manual_seed(11)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 48, 2, 8, 16, 4
    distinct = 6
    payload_dirs = torch.randn(distinct, value_dim, generator=generator)
    key_dirs = torch.randn(distinct, key_dim, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = payload_dirs[choice].double().cuda()
    write_keys = key_dirs[choice].double().cuda()
    query_keys = torch.randn(batch, tokens, heads, key_dim,
                             generator=generator).double().cuda()
    decay = torch.full((batch, tokens, heads), 0.95).double().cuda()

    hist_holder = {}
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    k = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)

    def f(p_, wk_, qk_, d_, s_, k_):
        ref = separated_sequential_oracle_scan(p_, wk_, qk_, d_, policy=POLICY,
                                               rank=rank, rank_max=rank,
                                               initial_state=s_, initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    gradcheck_ok = bool(torch.autograd.gradcheck(f, (p, wk, qk, d, s, k),
                                                 eps=1e-6, atol=1e-4, rtol=1e-4))
    hist = hist_holder["hist"]

    # 2. the floors and the kernel path's six gradients, on a gate-class shape
    # (fp32 inputs: the floor instrument's regime -- see registered_fixture).
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=23, batch=2, tokens=256, dtype=torch.float32)
    rank32, seg = 128, SEGMENT_TOKENS
    upstream = torch.randn(2, 256, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)).cuda()
    state = torch.zeros(2, HEADS, rank32, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(2, HEADS, rank32, dtype=torch.bool, device="cuda")
    keys = torch.zeros(2, HEADS, rank32, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    out = _k10_forward_launch(payload32, write_keys32, query_keys32, decay32,
                              state, live, keys, POL32, rank32, seg,
                              collect=True, warps=8, stages=3)
    # the oracle's autograd grads (fp32) and its fp64 twin (the floor)
    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        s_ = torch.zeros(2, HEADS, rank32, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(2, HEADS, rank32, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = separated_sequential_oracle_scan(p_.to(dtype), wk_.to(dtype),
                                               qk_.to(dtype), d_.to(dtype),
                                               policy=POLICY, rank=rank32,
                                               rank_max=rank32,
                                               initial_state=s_,
                                               initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return (p_.grad, wk_.grad, qk_.grad, d_.grad, s_.grad, k_.grad)

    oracle32 = oracle_grads(torch.float32)
    oracle64 = oracle_grads(torch.float64)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                             payload=out["payload"], op=out["op"],
                             live_after=out["live_after"])
    grads = _k10_backward_body(payload32, write_keys32, query_keys32, decay32,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank32, seg, POLICY.read_scale)
    names = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dstate",
             "dkeys")
    table = []
    for name, kern, orac, orac64 in zip(names, grads, oracle32, oracle64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        table.append(dict(output=name, floor=floor, deviation=deviation,
                         ratio_vs_floor=deviation / max(floor, 1e-30),
                         within_noise_factor=bool(deviation <= NOISE_FACTOR
                                                 * max(floor, 1e-6)
                                                 or deviation <= 1e-5)))
    # 3. the segment-recompute bit-equality on both channels.
    replay_ok = True
    from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward \
        import persistent_scan_trajectory
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_key_traj_kernel
    import triton
    decay_c = decay32.float().permute(0, 2, 1).contiguous()
    rows_count = 2 * HEADS
    for index in range(out["ckpt_state"].shape[0] - 1):
        zeros_w = torch.zeros(2, HEADS, SEGMENT_TOKENS, rank32,
                              dtype=torch.float32, device="cuda")
        _, exit_state, _exit_live, _traj = persistent_scan_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index], zeros_w,
            decay_c[:, :, index * SEGMENT_TOKENS:(index + 1) * SEGMENT_TOKENS],
            schedule.chunk(index * SEGMENT_TOKENS,
                           (index + 1) * SEGMENT_TOKENS))
        replay_ok &= bool(torch.equal(exit_state,
                                      out["ckpt_state"][index + 1]))
    ktraj = torch.empty((rows_count, SEGMENT_TOKENS, rank32, KEY_DIM),
                        dtype=torch.float32, device="cuda")
    _k10_key_traj_kernel[(rows_count,)](
        out["ckpt_keys"][0], write_keys32, schedule.slot, schedule.op, ktraj,
        SEGMENT_TOKENS, 256, 0, rank32, KEY_DIM,
        *write_keys32.stride(), H=HEADS, KP=triton.next_power_of_2(KEY_DIM),
        RP=triton.next_power_of_2(rank32), num_warps=4)
    replay_ok &= bool(torch.equal(ktraj[:, 0], out["ckpt_keys"][0].reshape(
        rows_count, rank32, KEY_DIM)))
    gate = dict(
        oracle_gradcheck_fp64=dict(ok=gradcheck_ok, branches=hist),
        floors_and_deviations=table,
        segment_recompute_bit_equal=bool(replay_ok),
        noise_factor=NOISE_FACTOR)
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """Block-call walls at the registered shape, against K6's banked walls."""
    payload, write_keys, query_keys, decay = registered_fixture()
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(BATCH, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")

    def forward_once():
        return _k10_forward_launch(payload, write_keys, query_keys, decay,
                                   state, live, keys, POL32, RANK,
                                   SEGMENT_TOKENS, collect=False, warps=8,
                                   stages=3)

    for _ in range(3):
        forward_once()                 # compile/warm
    torch.cuda.synchronize()
    repeats = 5
    started = time.monotonic()
    for _ in range(repeats):
        o = forward_once()
    torch.cuda.synchronize()
    fwd_ms = (time.monotonic() - started) / repeats * 1e3
    schedule = MatrixSchedule(slot=o["slot"], erase=o["erase"],
                              payload=o["payload"], op=o["op"],
                              live_after=o["live_after"])
    upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(4)).cuda()

    def backward_once():
        return _k10_backward_body(payload, write_keys, query_keys, decay,
                                  upstream, schedule, o["live_pre"],
                                  o["ckpt_state"], o["ckpt_live"],
                                  o["ckpt_keys"], RANK, SEGMENT_TOKENS,
                                  POLICY.read_scale)

    for _ in range(2):
        backward_once()
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(repeats):
        backward_once()
    torch.cuda.synchronize()
    bwd_ms = (time.monotonic() - started) / repeats * 1e3
    tok_per_scan = BATCH * TOKENS
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     segment_tokens=SEGMENT_TOKENS),
        forward_ms_per_scan=round(fwd_ms, 2),
        backward_ms_per_scan=round(bwd_ms, 2),
        forward_throughput_tok_per_s=round(tok_per_scan / (fwd_ms / 1e3)),
        k6_banked=dict(forward_ms_per_scan=26.5, backward_ms_per_scan=15.3),
        key_channel_cost_note=(
            "the key channel adds per token: two rank x key_dim defined-order "
            "folds and a key-norm fold against the value side's rank x "
            "value_dim folds (16 vs 128 wide), a rank x key_dim adjoint walk "
            "in the backward, and one key-table checkpoint per segment -- "
            "measured here against K6's banked walls, not estimated"),
        available=bool(separated_available(RANK, VALUE_DIM, KEY_DIM)))
    receipt["gate_d"] = gate
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "receipts" / "k10_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k10_separated_keys_gates",
                   registration=str(ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                   shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                              value_dim=VALUE_DIM, key_dim=KEY_DIM,
                              segment_tokens=SEGMENT_TOKENS),
                   policy=POLICY.record(), torch=torch.__version__)
    gate_a(receipt)
    gate_b(receipt)
    gate_d(receipt)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("gate_a", "gate_b", "gate_d")},
                     indent=1)[:4000])


if __name__ == "__main__":
    main()
