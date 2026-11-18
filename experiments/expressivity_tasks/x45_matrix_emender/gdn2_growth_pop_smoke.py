"""GDN2+GROWTH+POP feasibility smoke (PREREGISTRATION_GDN2_GROWTH_POP.md).

The POP lane's block-level feasibility gates, on the committed
``gdn2_growth_smoke.py`` pattern (imported verbatim -- the registrar, the
kernel path, the readout tail; GDN2's code is READ-ONLY throughout).  One
registered-shape GDN2 block (``GDN2ExternalMLPLayer(dim=576, expansion=2.,
head_dim=128, use_conv=True)``, 2 heads), the POP semantics under test =
HEAD-RECYCLING (state surgery only: ``S_h -> 0`` at a 64-token boundary when
the per-head saturation statistic crosses ``tau_pop``; parameters, shapes and
optimizer state untouched):

PS-0  wrapper parity: the POP lane's segmented real-pass forward with NO
      events (never-fires) vs the core module's own training forward -- the
      wrapper the arms install around the core must reproduce GDN2 verbatim
      when the triggers never fire, fp32 and under bf16 autocast.
PS-1  inertness at birth: the pop machinery PRESENT with an unreachable
      tau_pop (the trigger evaluated at every boundary, deciding nothing)
      is bit-identical to the growth-only walk (pop disabled == GDN2+growth,
      the registered birth receipt) -- across a mid-window growth event; and
      the same machinery at a reachable tau DOES fire (the perturbation is
      the capacity-freeing act, not a dead path).
PS-2  carry legality across a pop event: forced pops (incl. on a grown head,
      after a growth event): the popped head's boundary state is EXACTLY
      zero (the exact-capacity receipt -- the only exact zero in either
      machine's erasure vocabulary), the other heads' states are
      bit-identical to the un-popped walk (per-head isolation), the output
      is untouched before the first pop boundary, and the popped head
      refills from its own writes within one chunk (live keys).
PS-3  gradient legality: the pop-masked carry is differentiable and its
      leaf gradients are bit-equal to the equivalent explicit masked-carry
      construction; the popped head's pre-pop state receives EXACTLY zero
      gradient from post-pop segments (the exact-capacity backward); real
      optimizer steps through forced pops: finite loss/grads, the recycled
      head's parameters receive nonzero gradient at the first post-pop step
      (the re-entry receipt -- unlike the growth birth, a recycled head's
      parameters are live), and the pop touches NO parameter (the parameter
      set is unchanged; a pop is state-only surgery).
PS-4  kernel tolerance at the long-horizon shapes (T=8192 and T=16384, the
      regime's contexts): whole-window forward+backward at H in {2, 4, 8}
      plus growth+pop segment walks at the long shapes -- finite, walls.
PS-5  statistic anchors: the per-head participation ratio (the POP trigger
      statistic) and the mean-over-heads growth statistic from the real
      kernel's own boundary trajectory at init at T=8192 -- the first
      per-head anchors for the tau_pop / tau_sat calibration protocol.

Usage (card 6, this lane's leased card):

    eval "$(GPU_LEASE_VISIBLE=6 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 NDM_PIN_TRITON_AUTOTUNE=0 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_pop_smoke \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/gdn2_growth_pop_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

# The committed growth smoke, imported VERBATIM: the registrar (grow_layer),
# the kernel path (_kernel_inputs, _chunk_call), the readout tail (_readout),
# and the registered block shape.  Never edited.
from experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_smoke import (
    DIM, HEAD_DIM, HEAD_V_DIM, _chunk_call, _kernel_inputs, _readout,
    build_layer, grow_layer, mixer)

from ndm.models import external_gdn2

external_gdn2._load_gdn2_class()

ROOT = Path(__file__).resolve().parent
SEG = 64                      # the kernel's own chunk_size = the statistic boundary


# --------------------------------------------------------------------------
# The POP trigger bookkeeping (per-head saturation; the growth registration's
# statistic family, per-head instead of mean-over-heads)
# --------------------------------------------------------------------------

def head_pr(state: torch.Tensor) -> torch.Tensor:
    """Per-(batch, head) participation ratio PR(S) = (sum sigma)^2 / sum
    sigma^2 of the boundary state [B, H, K, V]; the empty state's PR is 0
    (an empty state is maximally UNsaturated -- the zero-state guard)."""
    sv = torch.linalg.svdvals(state.float())
    denom = (sv ** 2).sum(dim=-1)
    pr = (sv.sum(dim=-1) ** 2) / denom.clamp(min=1e-30)
    return torch.where(denom > 0, pr, torch.zeros_like(pr))   # [B, H]


class HeadPopTrigger:
    """Per-head saturation trigger, the registered candidate form: the
    batch-mean per-head PR, smoothed over the last ``window`` boundaries
    (reset on fire), firing per head at ``smoothed >= tau_pop``.  The
    head count follows the observed state (a growth event resets the
    bookkeeping at the new head count).  Write-pressure statistics gate
    nothing (the K8 idle-store receipt)."""

    def __init__(self, num_heads: int, tau_pop: float, window: int = 4):
        self.num_heads = num_heads
        self.tau_pop = tau_pop
        self.window = window
        self.history: list[torch.Tensor] = []   # per-boundary [H] batch means
        self.fires: list[dict] = []             # per-fire receipts

    def observe(self, boundary_state: torch.Tensor) -> list[int]:
        """One 64-token boundary: ``boundary_state`` [B, H, K, V] is the
        state at the boundary (before any surgery).  Returns the popped
        head indices (possibly empty) and records the receipts."""
        if boundary_state.shape[1] != self.num_heads:
            self.num_heads = boundary_state.shape[1]
            self.history = []
        pr = head_pr(boundary_state).mean(dim=0)               # [H] batch mean
        self.history.append(pr.detach().float().cpu())
        smoothed = torch.stack(self.history[-self.window:]).mean(dim=0)
        popped = [int(h) for h in range(self.num_heads)
                  if float(smoothed[h]) >= self.tau_pop]
        if popped:
            self.fires.append(dict(
                boundary=len(self.history), heads=popped,
                smoothed=[float(v) for v in smoothed],
                tau_pop=self.tau_pop))
            # reset the smoothed window on fire (the registered reset rule)
            self.history = []
        return popped


# --------------------------------------------------------------------------
# The 64-token segment walk: the growth event + the POP surgery, one code
# path for every arm (pop disabled == the growth-only walk, by construction)
# --------------------------------------------------------------------------

def segment_walk(old, grown, x, fire=None, pop_trigger=None, forced_pops=None,
                grad_mode=False, record_states=False):
    """The real pass of one block on one window: 64-token segments, the
    kernel's own chunk decomposition (the committed growth smoke's S-2
    receipt: bit-identical to the whole-window pass).

    ``old``/``grown``: GDN2ExternalMLPLayer modules (``grown=None``: no
    growth event).  ``fire``: the token index of the growth event (a
    64-boundary; the zero-padded carry + the grown module from there).
    ``pop_trigger``: a HeadPopTrigger evaluated on the walk's own boundary
    states (in the arms this plan comes from the probe pass; here the walk
    is its own probe) -- or None (pop disabled == GDN2+growth).
    ``forced_pops``: {chunk index -> [head indices]} forced pops for the
    legality gates, composed with the trigger's own decisions.

    Returns dict(o=layer-mixer readout walk [B,T,dim], states=the state
    entering each chunk from chunk 1 on (post-surgery; ``states[j]`` is the
    state entering chunk ``j+1``), pops=the surgery receipts, final_state).
    """
    core, gcore = mixer(old), (mixer(grown) if grown is not None else None)
    H_old = core.num_heads
    B, T, _ = x.shape
    n_chunks = T // SEG
    ctx = torch.enable_grad if grad_mode else torch.inference_mode
    with ctx():
        q, k, v, g, b, w = _kernel_inputs(core, x)
        if gcore is not None and fire is not None:
            gq, gk, gv, gg, gb, gw = _kernel_inputs(gcore, x)
        state = None
        outs = []
        states_at = []           # states[j] = the state entering chunk j+1
        pop_receipts = []
        for c in range(n_chunks):
            sl = slice(c * SEG, (c + 1) * SEG)
            grown_now = gcore is not None and fire is not None and c * SEG >= fire
            # -- the growth event at this boundary: zero-padded carry
            if gcore is not None and fire is not None and c * SEG == fire \
                    and state is not None:
                padded = torch.zeros(B, gcore.num_heads, HEAD_DIM, HEAD_V_DIM,
                                     dtype=state.dtype, device=state.device)
                padded[:, :H_old] = state
                state = padded
            # -- the POP surgery at this boundary (state-only, exact zero)
            heads_here = list(forced_pops.get(c, [])) if forced_pops else []
            if pop_trigger is not None and state is not None:
                heads_here = sorted(set(heads_here)
                                    | set(pop_trigger.observe(state)))
            if heads_here and state is not None:
                mask = torch.ones(state.shape[1], device=state.device)
                mask[heads_here] = 0.0
                state = state * mask.view(1, -1, 1, 1)
                pop_receipts.append(dict(chunk=c, heads=heads_here))
            if record_states and state is not None:
                states_at.append(state.detach().clone())
            # -- the segment's kernel call through the segment's own module
            if not grown_now:
                o_seg, state, _ = _chunk_call(
                    core, q[:, sl].contiguous(), k[:, sl].contiguous(),
                    v[:, sl].contiguous(), g[:, sl].contiguous(),
                    b[:, sl].contiguous(), w[:, sl].contiguous(),
                    initial_state=state, output_final_state=True)
                outs.append(_readout(core, o_seg, x[:, sl]))
            else:
                o_seg, state, _ = _chunk_call(
                    gcore, gq[:, sl].contiguous(), gk[:, sl].contiguous(),
                    gv[:, sl].contiguous(), gg[:, sl].contiguous(),
                    gb[:, sl].contiguous(), gw[:, sl].contiguous(),
                    initial_state=state, output_final_state=True)
                outs.append(_readout(gcore, o_seg, x[:, sl]))
    return dict(o=torch.cat(outs, dim=1), states=states_at,
                pops=pop_receipts, final_state=state)


def segment_walk_bf16(old, grown, x, **kw):
    """The same walk under the arms' bf16 autocast training regime."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        walk = segment_walk(old, grown, x, **kw)
        walk["o"] = walk["o"].float()
        walk["final_state"] = walk["final_state"].float()
        walk["states"] = [s.float() for s in walk["states"]]
    return walk


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------

def gate_ps0_wrapper_parity(receipt):
    """The never-fires wrapper path vs the core module's own forward (the
    wrapper the arms install replaces the core; with no events it must
    reproduce the core's training forward exactly)."""
    x = torch.randn(2, 2048, DIM, device="cuda")
    layer = build_layer(2, 3421).train()
    core = mixer(layer)
    walk = segment_walk(layer, None, x)                 # no events, fp32
    with torch.no_grad():
        o_core, _, _ = core(x, use_cache=False)         # the core's own path
    d_fp32 = float((walk["o"] - o_core).abs().max())
    walk_bf = segment_walk_bf16(layer, None, x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o_core_bf, _, _ = core(x, use_cache=False)
    d_bf16 = float((walk_bf["o"] - o_core_bf.float()).abs().max())
    receipt["ps0_wrapper_parity"] = {
        "fp32_max_abs_diff": d_fp32,
        "fp32_bit_identical": bool(torch.equal(walk["o"], o_core)),
        "bf16_max_abs_diff": d_bf16,
        "o_core_abs_max": float(o_core.abs().max()),
        "note": "the arms' wrapper (the 64-token segmented real pass, no "
                "events) vs the core's own training forward; the "
                "never-fires control reproduces GDN2 exactly (the growth "
                "smoke's S-2 receipt: the segmentation is the kernel's own "
                "chunk decomposition, bit-identical)",
    }
    d = receipt["ps0_wrapper_parity"]
    ok = d["fp32_max_abs_diff"] <= 1e-5 and d["bf16_max_abs_diff"] <= 5e-2
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_ps1_inertness(receipt):
    """Pop disabled (unreachable tau, trigger still evaluated) == GDN2+growth."""
    T, fire = 2048, 1024
    x = torch.randn(2, T, DIM, device="cuda")
    old = build_layer(2, 3422).eval()
    grown = grow_layer(old, 4, 3423, wake_up=True).eval()
    # (a) the growth-only walk (pop machinery absent) -- the GDN2+growth arm
    plain = segment_walk(old, grown, x, fire=fire)
    # (b) the same walk with the pop trigger PRESENT but unreachable
    trigger = HeadPopTrigger(2, tau_pop=float("inf"))
    armed = segment_walk(old, grown, x, fire=fire, pop_trigger=trigger)
    # (c) the same machinery at a REACHABLE tau (PR >= 1 fires at the first
    # boundary with a nonempty state): it must fire -- the perturbation is
    # the capacity-freeing act, not a dead path
    trigger_live = HeadPopTrigger(2, tau_pop=1.0)
    live = segment_walk(old, grown, x, fire=fire, pop_trigger=trigger_live)
    d = receipt["ps1_inertness"] = {
        "fp32_max_abs_diff_pop_disabled_vs_growth_only":
            float((plain["o"] - armed["o"]).abs().max()),
        "bit_identical": bool(torch.equal(plain["o"], armed["o"])),
        "trigger_decisions_unreachable_tau": trigger.fires,
        "trigger_fires_reachable_tau": len(trigger_live.fires),
        "live_walk_output_diff_vs_plain":
            float((plain["o"] - live["o"]).abs().max()),
        "note": "the pop-disabled arm IS GDN2+growth: with tau_pop "
                "unreachable the armed walk is bit-identical to the "
                "growth-only walk across a mid-window growth event",
    }
    ok = d["bit_identical"] and d["trigger_fires_reachable_tau"] > 0
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_ps2_pop_carry_legality(receipt):
    """Forced pops: exact zero, per-head isolation, refill, pre-pop output
    untouched; pops on a born-new grown head after a growth event."""
    T, fire = 2048, 1024
    x = torch.randn(2, T, DIM, device="cuda")
    old = build_layer(2, 3424).eval()
    grown = grow_layer(old, 4, 3425, wake_up=True).eval()
    # pops entering chunk 8 (head 1, pre-growth), chunk 20 (head 3, a
    # born-new grown head), chunk 28 (head 0, a shared head, post-growth)
    forced = {8: [1], 20: [3], 28: [0]}
    plain = segment_walk(old, grown, x, fire=fire, record_states=True)
    popped = segment_walk(old, grown, x, fire=fire, forced_pops=forced,
                          record_states=True)
    # states[j] is the state entering chunk j+1 (post-surgery at boundary
    # j+1); the surgery entering chunk b is states[b-1]
    exact_zeros = {}
    for bnd, heads in forced.items():
        for h in heads:
            exact_zeros[f"chunk_{bnd}_head_{h}_state_abs_max"] = float(
                popped["states"][bnd - 1][:, h].abs().max())
    # per-head isolation: head 2 (born new at the growth event) is never
    # popped -- its final state must be bit-identical to the plain walk
    iso = {
        "head_2_final_state_max_abs_diff": float(
            (popped["final_state"][:, 2] - plain["final_state"][:, 2]).abs().max()),
        "head_2_bit_identical": bool(torch.equal(
            popped["final_state"][:, 2], plain["final_state"][:, 2])),
    }
    # the output is untouched before the first pop boundary (chunk 8)
    pre = float((plain["o"][:, :8 * SEG] - popped["o"][:, :8 * SEG]).abs().max())
    # the popped head refills from its own writes within one chunk: the
    # state entering chunk bnd+1 (states[bnd]) has mass again
    refill = {}
    for bnd, heads in forced.items():
        for h in heads:
            refill[f"chunk_{bnd}_head_{h}_mass_one_chunk_after"] = float(
                popped["states"][bnd][:, h].norm())
    receipt["ps2_pop_carry_legality"] = {
        "forced_pops": {str(k): v for k, v in forced.items()},
        "popped_head_state_exact_zero": exact_zeros,
        "per_head_isolation": iso,
        "max_abs_output_diff_before_first_pop": pre,
        "popped_head_refill_frobenius": refill,
        "max_abs_output_diff_full_walk":
            float((plain["o"] - popped["o"]).abs().max()),
        "note": "exact zero at the surgery boundary (the exact-capacity "
                "receipt: GDN2's own erase/decay never reach zero at a "
                "step); per-head isolation by the kernel's construction; "
                "refill mass within one chunk (the recycled head's keys "
                "are live -- it writes into its virgin state immediately)",
    }
    d = receipt["ps2_pop_carry_legality"]
    ok = (all(v == 0.0 for v in exact_zeros.values())
          and iso["head_2_bit_identical"]
          and pre == 0.0
          and all(v > 0.0 for v in refill.values()))
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_ps3_gradient_legality(receipt):
    """Differentiability through the pop; the zero-flow backward; training
    through forced pops with live re-entry; no parameter/optimizer surgery."""
    receipt["ps3_gradient_legality"] = {}
    T = 512
    layer = build_layer(2, 3426).train()
    core = mixer(layer)

    def leaves_from(core, x):
        q, k, v, g, b, w = _kernel_inputs(core, x)
        g = g.detach()   # a fixed input in this check (the growth smoke's
        # own convention: sharing its projection subgraph across the two
        # backward passes would free it on the first)
        return [t.detach().clone().requires_grad_(True)
                for t in (q, k, v, b, w)], g

    # ---- (a) leaf-grad equality: the walk's masked carry vs the explicit
    # concatenate-a-zero-slice construction (identical math, must be equal)
    x = torch.randn(2, T, DIM, device="cuda")
    pop_chunk, pop_head = 4, 1
    leaf_grads = []
    for variant in ("walk_mask", "explicit_slice"):
        leaves, g = leaves_from(core, x)
        qs, ks, vs, bs, ws = leaves
        state = None
        outs = []
        for c in range(T // SEG):
            sl = slice(c * SEG, (c + 1) * SEG)
            if c == pop_chunk:
                if variant == "walk_mask":
                    mask = torch.ones(state.shape[1], device=state.device)
                    mask[pop_head] = 0.0
                    state = state * mask.view(1, -1, 1, 1)
                else:
                    state = torch.cat([
                        state[:, :pop_head],
                        torch.zeros_like(state[:, pop_head:pop_head + 1]),
                        state[:, pop_head + 1:]], dim=1)
            o_seg, state, _ = _chunk_call(
                core, qs[:, sl].contiguous(), ks[:, sl].contiguous(),
                vs[:, sl].contiguous(), g[:, sl].contiguous(),
                bs[:, sl].contiguous(), ws[:, sl].contiguous(),
                initial_state=state, output_final_state=True)
            outs.append(_readout(core, o_seg, x[:, sl]))
        torch.cat(outs, dim=1).pow(2).sum().backward()
        leaf_grads.append([float(t.grad.abs().max()) for t in leaves])
    grad_diff = max(abs(a - b) for a, b in zip(leaf_grads[0], leaf_grads[1]))
    receipt["ps3_gradient_legality"]["masked_carry_leaf_grads"] = {
        "walk_mask_max_abs": leaf_grads[0],
        "explicit_slice_max_abs": leaf_grads[1],
        "max_abs_diff_between_constructions": grad_diff,
    }

    # ---- (b) the zero-flow receipt: the popped head's PRE-POP state
    # receives exactly zero gradient from post-pop segments
    x2 = torch.randn(2, T, DIM, device="cuda")
    leaves, g = leaves_from(core, x2)
    qs, ks, vs, bs, ws = leaves
    state = None
    outs = []
    pre_state = None
    for c in range(T // SEG):
        sl = slice(c * SEG, (c + 1) * SEG)
        if c == pop_chunk:
            pre_state = state.detach().clone().requires_grad_(True)
            state = pre_state * torch.tensor(
                [1.0, 0.0], device=state.device).view(1, -1, 1, 1)
        o_seg, state, _ = _chunk_call(
            core, qs[:, sl].contiguous(), ks[:, sl].contiguous(),
            vs[:, sl].contiguous(), g[:, sl].contiguous(),
            bs[:, sl].contiguous(), ws[:, sl].contiguous(),
            initial_state=state, output_final_state=True)
        outs.append(_readout(core, o_seg, x2[:, sl]))
    torch.cat(outs, dim=1).pow(2).sum().backward()
    receipt["ps3_gradient_legality"]["zero_flow_backward"] = {
        "popped_head_prepop_state_grad_abs_max":
            float(pre_state.grad[:, pop_head].abs().max()),
        "other_head_prepop_state_grad_abs_max":
            float(pre_state.grad[:, 1 - pop_head].abs().max()),
        "note": "the pop kills the gradient flow from post-pop reads into "
                "the popped content (the exact-capacity backward -- the "
                "same semantics as our machine's POP S[j]=0); the other "
                "head's carry gradient flows normally",
    }

    # ---- (c) trains through forced pops; the recycled head re-enters live;
    # a pop touches NO parameter
    B = 4
    layer2 = build_layer(2, 3427).train()
    xt = torch.randn(B, T, DIM, device="cuda")
    target = torch.randn(B, T, DIM, device="cuda")
    opt = torch.optim.AdamW(layer2.parameters(), lr=1e-3)
    params_before = [(n, p.data_ptr()) for n, p in layer2.named_parameters()]
    losses, live_reentry = [], []
    grads_finite = True
    for step in range(4):
        # a live trigger at tau_pop = 1.0 fires at the first boundary with a
        # nonempty state in every window (PR >= 1 always) -- forced pops
        # every window, receipted per fire
        trigger = HeadPopTrigger(2, tau_pop=1.0)
        walk = segment_walk(layer2, None, xt, pop_trigger=trigger,
                            grad_mode=True)
        loss = F.mse_loss(walk["o"], target)
        opt.zero_grad()
        loss.backward()
        params = dict(layer2.named_parameters())
        live_reentry.append(dict(
            step=step, loss=float(loss.detach()),
            n_pop_fires=len(trigger.fires),
            # head 1 is the recycled head: its parameters are LIVE at the
            # first post-pop step (unlike the growth birth's one-step
            # wake-up -- a recycled head's keys/readout never stopped
            # being trained parameters)
            head1_k_proj_grad_abs_max=float(
                params["gdn2.gdn2.k_proj.weight"]
                .grad[HEAD_DIM:2 * HEAD_DIM].abs().max()),
            head1_o_proj_col_grad_abs_max=float(
                params["gdn2.gdn2.o_proj.weight"]
                .grad[:, HEAD_V_DIM:2 * HEAD_V_DIM].abs().max()),
        ))
        grads_finite = all(torch.isfinite(p.grad).all().item()
                           for p in layer2.parameters() if p.grad is not None)
        opt.step()
        losses.append(float(loss.detach()))
        if not grads_finite:
            break
    params_after = [(n, p.data_ptr()) for n, p in layer2.named_parameters()]
    receipt["ps3_gradient_legality"]["train_through_pops"] = {
        "losses": losses,
        "grads_finite": grads_finite,
        "live_reentry": live_reentry,
        "parameter_set_unchanged": params_before == params_after,
        "note": "a pop is state-only surgery: no parameter is created, "
                "removed or re-shaped and the optimizer state is untouched "
                "by construction (the receipt is the unchanged parameter "
                "set); the recycled head re-enters training live",
    }
    d = receipt["ps3_gradient_legality"]
    ok = (d["masked_carry_leaf_grads"]["max_abs_diff_between_constructions"] == 0.0
          and d["zero_flow_backward"]["popped_head_prepop_state_grad_abs_max"] == 0.0
          and d["zero_flow_backward"]["other_head_prepop_state_grad_abs_max"] > 0.0
          and grads_finite and all(math.isfinite(v) for v in losses)
          and params_before == params_after
          and all(r["n_pop_fires"] > 0 and
                  r["head1_k_proj_grad_abs_max"] > 0.0 and
                  r["head1_o_proj_col_grad_abs_max"] > 0.0
                  for r in live_reentry))
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_ps4_long_horizon_tolerance(receipt):
    """Fresh kernel-shape receipts at the regime's longer T."""
    receipt["ps4_long_horizon_tolerance"] = {}
    ok = True
    for T in (8192, 16384):
        for heads in (2, 4, 8):
            layer = build_layer(heads, 3428).train()
            x = torch.randn(2, T, DIM, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss = layer(x)[0].pow(2).sum()
            loss.backward()
            torch.cuda.synchronize()
            wall = time.time() - t0
            finite = all(torch.isfinite(p.grad).all().item()
                         for p in layer.parameters() if p.grad is not None)
            peak = torch.cuda.max_memory_allocated() / (1 << 30)
            receipt["ps4_long_horizon_tolerance"][f"T{T}_heads_{heads}"] = {
                "forward_backward_wall_s": round(wall, 3),
                "grads_finite": finite,
                "peak_allocated_gb": round(peak, 2),
            }
            ok = ok and finite
            del layer, x, loss
            torch.cuda.empty_cache()
        # the growth+pop machinery at the long shape: one full walk
        old = build_layer(2, 3429).eval()
        grown = grow_layer(old, 4, 3430, wake_up=True).eval()
        x = torch.randn(2, T, DIM, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        trigger = HeadPopTrigger(2, tau_pop=1.0)
        walk = segment_walk(old, grown, x, fire=T // 2, pop_trigger=trigger,
                           record_states=False)
        torch.cuda.synchronize()
        wall = time.time() - t0
        row = receipt["ps4_long_horizon_tolerance"][f"T{T}_growth_pop_walk"] = {
            "wall_s": round(wall, 3), "n_pop_fires": len(trigger.fires),
            "output_finite": bool(torch.isfinite(walk["o"]).all()),
            "final_state_finite": bool(
                torch.isfinite(walk["final_state"]).all()),
        }
        ok = ok and row["output_finite"] and row["final_state_finite"]
        del old, grown, x, walk
        torch.cuda.empty_cache()
    receipt["ps4_long_horizon_tolerance"]["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_ps5_statistic_anchors(receipt):
    """Per-head PR anchors at the long-horizon shape, at init: the tau_pop
    calibration's first real-module anchors (per-head granularity), plus the
    growth statistic (mean over heads) and the SVD wall."""
    T = 8192
    x = torch.randn(2, T, DIM, device="cuda")
    layer = build_layer(2, 3431).eval()
    core = mixer(layer)
    per_boundary = []
    t_svd = 0.0
    trigger = HeadPopTrigger(2, tau_pop=float("inf"))   # never fires: record
    with torch.inference_mode():
        q, k, v, g, b, w = _kernel_inputs(core, x)
        o, s_final, h = _chunk_call(core, q, k, v, g, b, w,
                                    output_final_state=True,
                                    return_intermediate_states=True)
        assert h.ndim == 5 and h.shape[1] == T // SEG, tuple(h.shape)
        for c in range(h.shape[1] - 1):
            t0 = time.time()
            pr = head_pr(h[:, c + 1])          # [B, H]
            torch.cuda.synchronize()
            t_svd += time.time() - t0
            trigger.observe(h[:, c + 1])
            per_boundary.append(dict(
                boundary=c + 1,
                batch_mean_per_head=[float(v) for v in pr.mean(dim=0)],
                mean_over_heads=float(pr.mean())))
    per_head_smoothed = []
    for hh in range(core.num_heads):
        series = [r["batch_mean_per_head"][hh] for r in per_boundary]
        smoothed = [sum(series[max(0, i - 3):i + 1]) /
                    len(series[max(0, i - 3):i + 1])
                    for i in range(len(series))]
        per_head_smoothed.append(dict(
            head=hh, smoothed_max=max(smoothed), smoothed_last=smoothed[-1],
            raw_max=max(series), raw_mean=sum(series) / len(series)))
    receipt["ps5_statistic_anchors"] = {
        "shape": {"batch": 2, "T": T, "heads": core.num_heads,
                  "d_k": HEAD_DIM, "boundaries": T // SEG},
        "per_head_smoothed_pr": per_head_smoothed,
        "growth_statistic_mean_over_heads": {
            "max": max(r["mean_over_heads"] for r in per_boundary),
            "last": per_boundary[-1]["mean_over_heads"],
            "tau_sat": 0.75 * HEAD_DIM,
        },
        "svd_wall_s_per_walk": round(t_svd, 3),
        "note": "at init, random inputs: the per-head PR (the POP trigger "
                "statistic) and the mean-over-heads PR (the growth trigger "
                "statistic) from the real kernel's own trajectory; the "
                "in-regime calibration on the C3 stream is the threshold's "
                "authority (the growth registration's disclosed protocol)",
        "verdict": "receipted (anchors, no pass/fail)",
    }
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts/gdn2_growth_pop_smoke.json")
    parser.add_argument("--seed", type=int, default=3421)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "the GDN2 chunk kernels are CUDA-only"
    torch.backends.cuda.matmul.allow_tf32 = False
    receipt = {
        "probe": "gdn2_growth_pop_smoke",
        "registration": "PREREGISTRATION_GDN2_GROWTH_POP.md (draft)",
        "config": {
            "layer": "GDN2ExternalMLPLayer(dim=576, expansion=2., "
                     "head_dim=128, use_conv=True) -- the registered 150M "
                     "arm's mixer shape",
            "pop_semantics": "HEAD-RECYCLING (state surgery only: S_h -> 0 "
                             "at 64-token boundaries on the per-head "
                             "saturation statistic; operator sign-off "
                             "pending -- escalated before the arms lock)",
            "trigger": "per-head participation ratio PR >= tau_pop "
                       "(registered candidate 0.75*d_k = 96, calibrated per "
                       "the growth registration's disclosed protocol), "
                       "smoothed over the last 4 boundaries, reset on fire",
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    gates = {}
    gates["PS-0 wrapper parity"] = gate_ps0_wrapper_parity(receipt)
    gates["PS-1 inertness at birth"] = gate_ps1_inertness(receipt)
    gates["PS-2 pop carry legality"] = gate_ps2_pop_carry_legality(receipt)
    gates["PS-3 gradient legality"] = gate_ps3_gradient_legality(receipt)
    gates["PS-4 long-horizon tolerance"] = gate_ps4_long_horizon_tolerance(receipt)
    gates["PS-5 statistic anchors"] = gate_ps5_statistic_anchors(receipt)
    receipt["gates"] = gates
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: ("met" if v else "NOT MET")
                      for k, v in gates.items()}, indent=1))
    failed = [k for k, v in gates.items() if not v]
    if failed:
        raise SystemExit(f"SMOKE GATES NOT MET: {failed}")
    print(f"all pop smoke gates met; receipt: {args.out}")


if __name__ == "__main__":
    main()
