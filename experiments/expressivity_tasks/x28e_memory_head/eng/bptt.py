"""Checkpoint/replay BPTT skeleton for the X28E slot-store tape head.

Pattern reference: ``ndm/triton/mlp_mem_fused.py`` — sparse checkpoints every
``CKPT_INTERVAL`` steps (``_mlp_mem_fwd_kernel``), reverse-replay per segment
(``_mlp_mem_bwd_kernel`` replays each segment forward then walks it in
reverse). Registration engineering gate 2: "Pop/decay/merge make the
recurrence NON-associative in slot space — no chunked/parallel scan; the
forward is a sequential per-token scan. Checkpoint payloads include slot
liveness/age/usage metadata, not just keys/values, so replay reproduces
allocation decisions exactly."

This phase-1 module is the EAGER skeleton (no Triton):

  * :func:`checkpoint_forward` — sequential scan saving a sparse checkpoint
    every ``CKPT_INTERVAL`` tokens. The payload includes the full slot store
    (keys, values, strength), the liveness/age/usage metadata, AND the
    discrete :class:`~eng.routing_trace.RoutingTrace` of the segment that
    starts at that boundary (replay-decision consistency gate, P1-6:
    "replay follows the trace — routing decisions are replay INPUTS, never
    recomputed from rounded state").
  * :func:`replay_forward` — segment-by-segment replay that FOLLOWS the
    trace, with a bf16-rounded-checkpoint mode proving trace-following
    survives rounding (X21E/X24 precision receipt: fp32 live state, bf16
    replay checkpoints, fp32 parameters). Boundary reports compare the
    replayed store against the forward checkpoints; liveness/age must match
    exactly ("reproduces allocation decisions exactly").
  * :func:`replay_consistency` — the P1-6 multi-checkpoint branch-equivalence
    gate: replayed routing == forward trace at every checkpoint boundary.
  * :func:`replay_gradients` — reverse-segment gradient skeleton (the eager
    form of the reverse-replay BPTT the phase-2 backward kernel implements):
    replay each segment under autograd in reverse order threading the state
    gradient, exactly as ``_mlp_mem_bwd_kernel`` threads ``dtheta``.

FLAGGED residual (RESOLVED pre-M2 by supervisor resolution G10, binding):
the free-by-decay death check (strength <= free_threshold) is now TRACED
(int8 codes per token in ``RoutingTrace.freed``) and replay consumes the
traced frees instead of recomputing them from (possibly bf16-rounded)
replayed strengths — any architectural decision near a threshold can flip
under rounding, and death decisions are no exception. The replay-
consistency gate covers the freed codes via ``assert_traces_equal``.

CKPT_INTERVAL: placeholder default 16 (TO-BE-BOUND-AT-FREEZE, per the
registration "CKPT_INTERVAL bound at commit").
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .tape_head import TapeHeadConfig, TapeState, init_state, tape_scan
from .routing_trace import RoutingTrace, assert_traces_equal

DEFAULT_CKPT_INTERVAL = 16


# ---------------------------------------------------------------------------
# Checkpoint payloads.
# ---------------------------------------------------------------------------
def _payload(state: TapeState, t: int, trace_segment: Optional[RoutingTrace],
             dtype: torch.dtype) -> Dict:
    """Checkpoint payload: full slot store + bookkeeping + segment trace.

    keys/values/strength are stored in ``dtype`` (fp32 exact mode, or bf16
    for the declared bf16-rounded-checkpoint replay mode); live (bool) and
    age (int64) are stored EXACTLY in every mode so replay reproduces
    allocation decisions exactly; usage is bookkeeping (fp32).
    """
    return {
        "t": t,
        "keys": state.keys.detach().to(dtype).clone(),
        "values": state.values.detach().to(dtype).clone(),
        "strength": state.strength.detach().to(dtype).clone(),
        "live": state.live.detach().clone(),
        "age": state.age.detach().clone(),
        "usage": state.usage.detach().clone(),
        "trace_segment": trace_segment,
        "dtype": dtype,
    }


def _load_checkpoint(payload: Dict, requires_grad: bool = False) -> TapeState:
    state = TapeState(
        keys=payload["keys"].to(torch.float32),
        values=payload["values"].to(torch.float32),
        strength=payload["strength"].to(torch.float32),
        live=payload["live"].to(torch.bool),
        age=payload["age"].to(torch.int64),
        usage=payload["usage"].to(torch.float32),
    )
    if requires_grad:
        state.keys.requires_grad_(True)
        state.values.requires_grad_(True)
        state.strength.requires_grad_(True)
    return state


def _boundary_diff(replayed: TapeState, payload: Dict) -> Dict:
    """Compare a replayed boundary state against the forward checkpoint."""
    ref = _load_checkpoint(payload)
    report: Dict = {"t": payload["t"]}
    for name, a, b in (("keys", replayed.keys, ref.keys),
                       ("values", replayed.values, ref.values),
                       ("strength", replayed.strength, ref.strength)):
        diff = (a - b).abs()
        denom = b.abs().clamp_min(1e-6)
        report[f"{name}_max_abs"] = float(diff.max())
        report[f"{name}_max_rel"] = float((diff / denom).max())
    report["live_equal"] = bool(torch.equal(replayed.live, ref.live))
    report["age_equal"] = bool(torch.equal(replayed.age, ref.age))
    return report


# ---------------------------------------------------------------------------
# Forward with sparse checkpoints.
# ---------------------------------------------------------------------------
def checkpoint_forward(k, q, v, write_strength, pop_strength, append_logit,
                       state0: TapeState, cfg: TapeHeadConfig,
                       ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                       checkpoint_dtype: torch.dtype = torch.bfloat16,
                       append_err_coef=None):
    """Sequential forward scan with sparse checkpoints every ``ckpt_interval``
    tokens (plus a boundary-0 checkpoint of the entry state).

    ``checkpoints[i]`` holds the store at token ``t_i`` TOGETHER WITH the
    routing trace of the segment that starts at ``t_i`` — replay of segment
    ``i`` loads ``checkpoints[i]`` and follows ``checkpoints[i][
    "trace_segment"]``.

    Returns ``(read_out [B,T,NH,V], final_state (fp32), trace, checkpoints)``.
    """
    T = k.shape[1]
    assert 1 <= ckpt_interval
    checkpoints: List[Dict] = [_payload(state0, 0, None, checkpoint_dtype)]
    state = state0
    outs: List[torch.Tensor] = []
    traces: List[RoutingTrace] = []
    t0 = 0
    while t0 < T:
        t1 = min(t0 + ckpt_interval, T)
        out_seg, state, tr_seg = tape_scan(
            k[:, t0:t1], q[:, t0:t1], v[:, t0:t1],
            write_strength[:, t0:t1], pop_strength[:, t0:t1], append_logit[:, t0:t1],
            state, cfg, append_err_coef=append_err_coef, trace=None, collect_trace=True)
        # the segment trace belongs to the checkpoint this segment starts from
        checkpoints[-1]["trace_segment"] = tr_seg
        outs.append(out_seg)
        traces.append(tr_seg)
        checkpoints.append(_payload(state, t1, None, checkpoint_dtype))
        t0 = t1
    read_out = torch.cat(outs, dim=1)
    trace = RoutingTrace.cat(traces)
    return read_out, state, trace, checkpoints


# ---------------------------------------------------------------------------
# Trace-following replay.
# ---------------------------------------------------------------------------
def replay_forward(k, q, v, write_strength, pop_strength, append_logit,
                   checkpoints: List[Dict], cfg: TapeHeadConfig,
                   ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                   append_err_coef=None):
    """Replay the forward scan segment by segment, FOLLOWING THE TRACE.

    Routing decisions (case, write/correction target, pop fire + target,
    append allocation target, W-ON) are replay INPUTS taken from each
    checkpoint's trace segment — never recomputed from (possibly bf16-
    rounded) state. Magnitudes (read mixtures, gate strengths, delta-rule
    corrections, strength updates) are recomputed in fp32 from the replayed
    state.

    Returns ``(read_out, replayed_trace, boundary_reports)`` where each
    boundary report compares the replayed store at the segment boundary with
    the forward checkpoint at the same boundary.
    """
    T = k.shape[1]
    num_seg = len(checkpoints) - 1
    assert num_seg == (T + ckpt_interval - 1) // ckpt_interval
    outs: List[torch.Tensor] = []
    traces: List[RoutingTrace] = []
    reports: List[Dict] = []
    for i in range(num_seg):
        t0 = checkpoints[i]["t"]
        t1 = checkpoints[i + 1]["t"]
        seg_trace = checkpoints[i]["trace_segment"]
        state = _load_checkpoint(checkpoints[i], requires_grad=False)
        out_seg, state_r, tr_seg = tape_scan(
            k[:, t0:t1], q[:, t0:t1], v[:, t0:t1],
            write_strength[:, t0:t1], pop_strength[:, t0:t1], append_logit[:, t0:t1],
            state, cfg, append_err_coef=append_err_coef, trace=seg_trace,
            collect_trace=True)
        outs.append(out_seg)
        traces.append(tr_seg)
        reports.append(_boundary_diff(state_r, checkpoints[i + 1]))
    read_out = torch.cat(outs, dim=1)
    trace = RoutingTrace.cat(traces)
    return read_out, trace, reports


def replay_consistency(k, q, v, write_strength, pop_strength, append_logit,
                       state0: TapeState, cfg: TapeHeadConfig,
                       ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                       checkpoint_dtype: torch.dtype = torch.bfloat16,
                       append_err_coef=None) -> Dict:
    """P1-6 multi-checkpoint branch-equivalence gate.

    Runs the checkpointed forward, replays following the trace, and receipts:
    ``traces_equal`` (replayed routing == forward trace at every checkpoint
    boundary — trivially per-segment here, but it proves the replay plumbing
    consumes the trace), per-case coverage counts, boundary state diffs, and
    liveness/age exactness. In bf16 mode this proves trace-following
    survives checkpoint rounding.
    """
    read_out, _, trace, checkpoints = checkpoint_forward(
        k, q, v, write_strength, pop_strength, append_logit, state0, cfg,
        ckpt_interval=ckpt_interval, checkpoint_dtype=checkpoint_dtype,
        append_err_coef=append_err_coef)
    replay_out, replay_trace, reports = replay_forward(
        k, q, v, write_strength, pop_strength, append_logit, checkpoints, cfg,
        ckpt_interval=ckpt_interval,
        append_err_coef=append_err_coef)
    gate: Dict = {}
    try:
        assert_traces_equal(trace, replay_trace, what="P1-6 replay consistency")
        gate["traces_equal"] = True
    except AssertionError as e:
        gate["traces_equal"] = False
        gate["traces_equal_error"] = str(e)
    gate["forward_case_counts"] = trace.counts()
    gate["boundary_reports"] = reports
    gate["live_and_age_exact"] = all(r["live_equal"] and r["age_equal"] for r in reports)
    out_diff = (replay_out - read_out).abs()
    gate["output_max_abs"] = float(out_diff.max())
    denom = read_out.abs().clamp_min(1e-6)
    gate["output_max_rel"] = float((out_diff / denom).max())
    return gate


# ---------------------------------------------------------------------------
# Reverse-replay gradient skeleton (eager form of the phase-2 bwd kernel).
# ---------------------------------------------------------------------------
def replay_gradients(k, q, v, write_strength, pop_strength, append_logit,
                     checkpoints: List[Dict], cfg: TapeHeadConfig,
                     ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                     d_out: Optional[torch.Tensor] = None,
                     d_final_state: Optional[Dict[str, torch.Tensor]] = None,
                     append_err_coef=None) -> Dict[str, torch.Tensor]:
    """Reverse-replay BPTT skeleton: replay segments in reverse under
    autograd, threading the state gradient from segment to segment, exactly
    the ``_mlp_mem_bwd_kernel`` pattern (reload segment-entry checkpoint,
    replay forward, walk in reverse threading the state gradient).

    In fp32-checkpoint mode this reproduces the direct full-graph autograd
    gradients (receipted in the phase-1 tests); in bf16-checkpoint mode it
    is the declared rounded-replay gradient path (phase-2 gate tolerances
    apply: input/parameter gradient max-relative error < 0.08).

    ``d_out``: [B, T, NH, V] gradient seed w.r.t. the read outputs (zeros if
    None). ``d_final_state``: optional grads of a downstream loss w.r.t. the
    FINAL store's keys/values/strength.

    Returns per-input gradients, each shaped like the input.
    """
    T = k.shape[1]
    num_seg = len(checkpoints) - 1
    grads = {
        "k": torch.zeros_like(k, dtype=torch.float32),
        "q": torch.zeros_like(q, dtype=torch.float32),
        "v": torch.zeros_like(v, dtype=torch.float32),
        "write_strength": torch.zeros_like(write_strength, dtype=torch.float32),
        "pop_strength": torch.zeros_like(pop_strength, dtype=torch.float32),
        "append_logit": torch.zeros_like(append_logit, dtype=torch.float32),
    }
    if d_out is None:
        d_out = torch.zeros(k.shape[0], T, k.shape[2], v.shape[-1],
                            device=k.device, dtype=torch.float32)

    # incoming state-grads for the boundary AFTER the last replayed segment
    g_keys = None if d_final_state is None else d_final_state.get("keys")
    g_values = None if d_final_state is None else d_final_state.get("values")
    g_strength = None if d_final_state is None else d_final_state.get("strength")

    for i in reversed(range(num_seg)):
        t0, t1 = checkpoints[i]["t"], checkpoints[i + 1]["t"]
        state = _load_checkpoint(checkpoints[i], requires_grad=True)
        seg_inputs = [k[:, t0:t1], q[:, t0:t1], v[:, t0:t1],
                      write_strength[:, t0:t1], pop_strength[:, t0:t1],
                      append_logit[:, t0:t1]]
        out_seg, state_out, _ = tape_scan(
            *seg_inputs, state, cfg, append_err_coef=append_err_coef,
            trace=checkpoints[i]["trace_segment"], collect_trace=False)
        z_out = torch.zeros_like(out_seg)
        gk = torch.zeros_like(state.keys) if g_keys is None else g_keys
        gv = torch.zeros_like(state.values) if g_values is None else g_values
        gs = torch.zeros_like(state.strength) if g_strength is None else g_strength
        seg_grads = torch.autograd.grad(
            outputs=[out_seg, state_out.keys, state_out.values, state_out.strength],
            grad_outputs=[d_out[:, t0:t1], gk, gv, gs],
            inputs=list(seg_inputs) + [state.keys, state.values, state.strength],
            allow_unused=True)
        names = ["k", "q", "v", "write_strength", "pop_strength", "append_logit"]
        for name, g in zip(names, seg_grads[:6]):
            if g is not None:
                grads[name][:, t0:t1] = grads[name][:, t0:t1] + g
        # thread the state gradient to the previous (earlier) segment
        g_keys, g_values, g_strength = seg_grads[6], seg_grads[7], seg_grads[8]
    return grads


__all__ = [
    "DEFAULT_CKPT_INTERVAL", "checkpoint_forward", "replay_forward",
    "replay_consistency", "replay_gradients",
]
