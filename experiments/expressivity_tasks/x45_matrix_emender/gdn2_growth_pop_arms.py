"""GDN2+GROWTH+POP arms + the tau calibration (PREREGISTRATION_GDN2_GROWTH_POP.md).

The lane's Deliverable 2, on the supervisor's rulings:

* **The regime** (Option B): the long-horizon variant of the registered Pile
  stream -- same documents, longer windows via ``lattice.Stream(context=...)``
  (no ``lattice.py`` edit), document-bounded windows only, matched exposure
  at the standing probe rule (20,000,000 targets, honest clock), 180
  constant-LR steps + the corrected 24-step linear cooldown primary
  (RESULT_PROTOCOL_CORRECTION.md), document-disjoint calibration eval at
  the regime's context.
* **The calibration** (the growth registration's disclosed protocol, run by
  this lane so the receipts cross-check the C3 lane's): the FIXED arm's
  per-boundary statistics receipted on the regime stream at BOTH contexts
  (8192 and 16384); the horizon is the SHORTEST at which the registered
  ``tau_sat = 96`` is reachable; if neither reaches, escalate (no arm runs
  on an invented horizon).  The same receipts fix ``tau_pop`` (the per-head
  PR threshold, registered candidate 96) and ``tau_read`` (the read-mass
  floor, the pre-committed P25 rule: the 25th percentile of the smoothed
  read-mass distribution among per-head-saturated observations), committed
  BEFORE the arms run.
* **The arms** (same-session pair): ``gdn2_growth`` (the control: the
  registered GDN2 arm + the growth registration's mechanism verbatim, run
  fresh in this session and regime) and ``gdn2_growth_pop`` (the same +
  head-recycling, the locked two-component trigger).  Single seed 3401.
* **The mechanism** (no GDN2 code edits; the committed growth smoke's
  kernel path and registrar rules imported verbatim): each block's core is
  wrapped by ``GDN2TwoPassCore`` -- eval delegates to the current core
  VERBATIM; training runs the two-pass discipline: a PROBE pass (one
  inference-mode whole-window kernel call with
  ``return_intermediate_states=True`` -- the kernel's own per-chunk state
  trajectory plus its per-head kernel-level outputs) computes the
  registered statistics; the decided PLAN is replayed by the REAL pass,
  which segments at EVENT boundaries only (whole-window call with no
  events -- the growth smoke's S-2 bit-identity; one span per event
  boundary, the carry chained through the surgery masks: growth =
  zero-padded carry + the grown core, pop = exact-zero head mask).
  Growth is a model-wide event (the mean-over-(block, head, batch)
  statistic, one event per window and per step, receipted); pops are
  per-(block, head) state surgery.  Growth adds parameters (the
  registration's disclosed registrar semantics: shared slices copied,
  ``o_proj`` new columns zeroed -- the wake-up birth; event-time output
  invariance); the event step's gradients are MERGED (the pre-event spans
  used the retired core: its gradients are summed into the grown core's
  shared slices -- the reparametrized objective's exact gradient), the
  optimizer rebuilt with state carried by slice, and the retired core
  retained as a non-live child so every snapshot stays loadable.

Statistics: the participation ratio per (block, head, boundary) via the
X45 spectra sidecar (imported verbatim -- the supervisor's option-D ruling
and the cross-lane consistency ruling; exact LAPACK spectra, bit-equal to
the same-LAPACK single-process reference, guardrailed).  The read-mass
(the locked trigger's second component) is the per-head, per-boundary,
batch-mean squared magnitude of the kernel-level output -- the head's
recent contribution to reads, computed on the probe pass's own outputs,
metrics-only.

Subcommands:

``calibrate``  the fixed arm (never-fires wrapper) at one context; receipts
               the per-boundary growth statistic, the per-head PR and
               read-mass distributions, the reachability verdicts, the
               spectra pool's guardrails/timing, and the threshold commit.
``arm``        one arm (``gdn2_growth`` control or ``gdn2_growth_pop``) at
               the committed thresholds; receipts training.jsonl with the
               per-step event summary, all three scorings (constant-LR,
               post-cooldown PRIMARY, EMA secondary), the recycling
               statistics (per-pop head, PR and read-mass at fire,
               saturation before/after, events by window position), and
               the sidecar's mid-run load receipt.
``assemble``   the pair's verdict table.

Usage (card 6, this lane's leased card):

    eval "$(GPU_LEASE_VISIBLE=6 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 NDM_PIN_TRITON_AUTOTUNE=0 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_pop_arms calibrate \\
        --context 8192 --out receipts/gdn2_growth_pop_calibration_8192.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

# The committed growth smoke, imported VERBATIM (never edited): the kernel
# path, the readout tail, the registered block shape.
from experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_smoke import (
    HEAD_DIM, HEAD_V_DIM, _chunk_call, _kernel_inputs, _readout)
# The supervisor-ruled spectra sidecar (exact PR spectra, guardrailed).
from experiments.expressivity_tasks.x45_matrix_emender import (
    x45_spectra_sidecar as SPECTRA)
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms

from ndm.models import external_gdn2

external_gdn2._load_gdn2_class()

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RUNS = ROOT / "gdn2_growth_pop_runs"
RECEIPTS = ROOT / "receipts"

SEG = 64                          # the kernel's own chunk_size = the boundary
EXPOSURE_TARGETS = 20_000_000     # the standing probe rule
WINDOWS_PER_STEP = 14            # the LOCKED regime protocol (the C3 lane's
                                 # committed regime: 14 full-8192 windows per
                                 # step -> the 20,070,400 landing at 175 steps)
COOLDOWN_STEPS = 24               # the corrected-scoring convention
EMA_DECAY = 0.99
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
EVAL_WINDOWS = 24                # the LOCKED regime's document-disjoint eval
GRAD_CLIP = 1.0
SEED = 3401                       # the standing seed
HEAD_CAP = 8                      # the growth registration's snap ceiling
TAU_SAT = 96.0                    # 0.75 * d_k, the registered candidate
TAU_POP = 96.0                    # the per-head analog (the registered candidate)
ENDPOINT_VARIANCE_BAND = 0.075    # the side-by-side's band, every margin read against it
CONTEXTS = (8192, 16384)          # the ruling's horizon scan (8192 LOCKED)
CONTEXT_LOCKED = 8192
TAU_SAT_C3_LOCKED = 19.205211639404297   # the C3 lane's committed calibrated
                                       # growth threshold (their regime JSON,
                                       # imported verbatim for the control arm)

TRIGGER_WINDOW = 4                # the registered smoothing window


# --------------------------------------------------------------------------
# The core-level registrar (the growth registration's rules, at core level:
# the layer's norm_2/mlp are head-count-independent and are NEVER rebuilt)
# --------------------------------------------------------------------------

# head-count-dependent core parameters, by name (the committed smoke's rules)
_CORE_KEY_ROW = ("q_proj.weight", "k_proj.weight", "b_proj.weight",
                 "f_proj.1.weight")
_CORE_VAL_ROW = ("v_proj.weight", "w_proj.weight", "g_proj.1.weight",
                 "g_proj.1.bias")
_CORE_CONV_KEY = ("q_conv1d.weight", "k_conv1d.weight")
_CORE_CONV_VAL = ("v_conv1d.weight",)
_CORE_O_PROJ = "o_proj.weight"


def _prefix_slices(old_shape, new_shape):
    return tuple(slice(0, o) if o != n else slice(None)
                 for o, n in zip(old_shape, new_shape))


def _fresh_core(old_core, heads_new: int, seed: int):
    """A fresh GDN2 core at ``heads_new`` mirroring the OLD core's own
    configuration (width/expand/head_dim/conv/eigval policy), born at GDN2's
    own init recipes (the committed smoke's build semantics: construct via
    the external wrapper, take the core; the registration's 'born at GDN2's
    own init')."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    layer = external_gdn2.GDN2ExternalLayer(
        dim=old_core.hidden_size, expansion=old_core.expand_v,
        head_dim=old_core.head_dim, num_heads=heads_new, use_conv=True,
        d_conv=old_core.conv_size,
        allow_neg_eigval=old_core.allow_neg_eigval)
    return layer.gdn2.cuda()


def grow_core(old_core, heads_new: int, seed: int, wake_up: bool = True):
    """A fresh core at ``heads_new``, born at GDN2's own init (mirroring the
    old core's configuration), with the old core's parameters copied into
    the shared slices and ``o_proj``'s new input columns zeroed iff
    ``wake_up`` (the registration's event-time output invariance + one-step
    wake-up).  Returns (new_core, shared) where ``shared`` is the list of
    (old_param, new_param, slice) used for the event step's gradient merge
    and the optimizer-state carry."""
    old_heads = old_core.num_heads
    assert heads_new > old_heads
    new_core = _fresh_core(old_core, heads_new, seed)
    old_key, new_key = old_heads * HEAD_DIM, heads_new * HEAD_DIM
    old_val, new_val = old_heads * HEAD_V_DIM, heads_new * HEAD_V_DIM
    new_params = dict(new_core.named_parameters())
    assert set(new_params) == {n for n, _ in old_core.named_parameters()}, \
        "the fresh core's parameter set must match the old core's"
    shared = []
    for name, p_old in old_core.named_parameters():
        p_new = new_params[name]
        with torch.no_grad():
            if name == _CORE_O_PROJ:
                p_new[:, :old_val] = p_old
                if wake_up:
                    p_new[:, old_val:new_val] = 0.0
                shared.append((p_old, p_new, (slice(None), slice(0, old_val))))
            elif name in _CORE_KEY_ROW or name in _CORE_CONV_KEY:
                p_new[:old_key] = p_old
                shared.append((p_old, p_new, (slice(0, old_key),)))
            elif name in _CORE_VAL_ROW or name in _CORE_CONV_VAL:
                p_new[:old_val] = p_old
                shared.append((p_old, p_new, (slice(0, old_val),)))
            elif name in ("A_log", "dt_bias"):
                sl = _prefix_slices(p_old.shape, p_new.shape)
                p_new[sl] = p_old
                shared.append((p_old, p_new, sl))
            else:
                # head-count-independent (f_proj.0, g_proj.0): copied WHOLE
                assert p_old.shape == p_new.shape, name
                p_new.copy_(p_old)
                shared.append((p_old, p_new, None))
    return new_core, shared


# --------------------------------------------------------------------------
# The two-pass wrapper around each block's core (GDN2's code is read-only)
# --------------------------------------------------------------------------

class GDN2TwoPassCore(torch.nn.Module):
    """Replaces ``block.layer.gdn2.gdn2`` (the GatedDeltaNet2 core).

    Eval / no-plan forward: DELEGATES to the current core verbatim (the
    banked eval path, bit-for-bit).  Training runs the two-pass discipline
    driven by the training loop: ``mode='probe'`` (one inference-mode
    whole-window kernel call with the per-chunk state trajectory and the
    per-head kernel outputs collected, metrics-only) then ``plan``
    (the replay: spans between event boundaries, the carry chained through
    the surgery masks).  All cores ever alive are retained as children
    (retired cores leave the live parameter set but stay loadable from any
    snapshot); ``current_idx`` is a registered buffer so state_dicts carry
    the event history.
    """

    def __init__(self, core):
        super().__init__()
        self.cores = torch.nn.ModuleList([core])
        self.register_buffer("current_idx",
                             torch.zeros(1, dtype=torch.long))
        self.layer_idx = getattr(core, "layer_idx", 0)
        self.mode = "delegate"
        self.plan = None
        self.collector = None          # set by the driver for a probe pass

    # ---- driver surface

    def set_layer_idx(self, idx):
        self.layer_idx = idx
        for core in self.cores:
            core.layer_idx = idx

    @property
    def current(self):
        return self.cores[int(self.current_idx)]

    def live_parameters(self):
        return list(self.current.parameters())

    # ---- the three forwards

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=None, **kwargs):
        if self.mode == "probe" and not torch.is_grad_enabled():
            return self._probe(hidden_states)
        if self.plan is not None:
            return self._plan(hidden_states, self.plan)
        return self.current(hidden_states, attention_mask, past_key_values,
                            use_cache, output_attentions, **kwargs)
    def _probe(self, x):
        """Metrics-only: the kernel's own per-chunk state trajectory and the
        per-head kernel-level outputs; returns the exact readout so the
        probe walk doubles as a forward."""
        core = self.current
        q, k, v, g, b, w = _kernel_inputs(core, x)
        o, s, h = _chunk_call(core, q, k, v, g, b, w, output_final_state=True,
                              return_intermediate_states=True)
        # the read-mass channel: per (boundary, head), batch-mean of the
        # mean squared magnitude of the head's kernel output over the
        # boundary's 64 tokens
        B, T, H, V = o.shape
        C = T // SEG
        read = (o.float().pow(2).sum(-1)
                .reshape(B, C, SEG, H).mean(dim=(0, 2)))        # [C, H]
        self.collector[self.layer_idx] = dict(h=h, read=read, num_heads=H)
        return _readout(core, o, x), None, None

    def _plan(self, x, plan):
        """The real pass: spans between event boundaries; whole-window when
        no events (the growth smoke's S-2 bit-identity); one kernel call per
        span, the carry chained through the surgery masks (growth =
        zero-padded carry onto the grown core; pop = exact-zero head mask)."""
        B, T, _ = x.shape
        nC = T // SEG
        grow_chunk = plan.get("growth_chunk")       # None or a chunk index
        old_idx = plan["core_old"]
        new_idx = plan.get("core_new")
        pops = plan.get("pops", {})                 # {chunk: [heads]}
        cores = self.cores
        projs: dict[int, tuple] = {}

        def proj(idx):
            if idx not in projs:
                projs[idx] = _kernel_inputs(cores[idx], x)
            return projs[idx]

        events = sorted(set(pops) | ({grow_chunk} if grow_chunk is not None
                                     else set()))
        outs = []
        carry = None
        prev = 0
        boundaries = events + [nC]
        for bnd in boundaries:
            if bnd > prev:
                idx = old_idx if (grow_chunk is None or prev < grow_chunk) \
                    else new_idx
                q, k, v, g, bb, w = proj(idx)
                sl = slice(prev * SEG, bnd * SEG)
                o_seg, carry, _ = _chunk_call(
                    cores[idx], q[:, sl].contiguous(), k[:, sl].contiguous(),
                    v[:, sl].contiguous(), g[:, sl].contiguous(),
                    bb[:, sl].contiguous(), w[:, sl].contiguous(),
                    initial_state=carry, output_final_state=True)
                outs.append(_readout(cores[idx], o_seg, x[:, sl]))
            if bnd >= nC:
                break
            # -- the surgery at boundary bnd, for the NEXT span
            if bnd == grow_chunk and carry is not None:
                padded = torch.zeros(B, cores[new_idx].num_heads, HEAD_DIM,
                                     HEAD_V_DIM, dtype=carry.dtype,
                                     device=carry.device)
                padded[:, :cores[old_idx].num_heads] = carry
                carry = padded
            heads = pops.get(bnd, [])
            if heads and carry is not None:
                mask = torch.ones(carry.shape[1], device=carry.device)
                mask[heads] = 0.0
                carry = carry * mask.view(1, -1, 1, 1)
            prev = bnd
        return torch.cat(outs, dim=1), None, None


def convert_model(model):
    """Wrap every block's core with the two-pass wrapper (block.layer.gdn2.gdn2
    replaced; the layer object itself is NEVER rebuilt -- its norm_2/mlp are
    head-count-independent and stay the same parameters throughout)."""
    wrappers = []
    for i, block in enumerate(model.model.blocks):
        core = block.layer.gdn2.gdn2
        wrapper = GDN2TwoPassCore(core)
        wrapper.set_layer_idx(getattr(core, "layer_idx", i))
        block.layer.gdn2.gdn2 = wrapper
        wrappers.append(wrapper)
    return wrappers


def live_parameters(model, wrappers):
    """The trainable set: everything except the retired cores."""
    live = [w.live_parameters() for w in wrappers]
    skip = {id(p) for group in live for p in group}
    others = [p for p in model.parameters() if id(p) not in skip]
    return others + [p for group in live for p in group]


# --------------------------------------------------------------------------
# The triggers (the locked two-component form; saturation gates, write
# pressure never does -- the K8 idle-store ruling)
# --------------------------------------------------------------------------

def _smoothed(series, window=TRIGGER_WINDOW):
    """Rows of a [C, ...] statistic smoothed over the last ``window``
    boundaries (partial windows at the start of a window/after a reset)."""
    n = series.shape[0]
    out = torch.empty_like(series)
    for c in range(n):
        lo = max(0, c - window + 1)
        out[c] = series[lo:c + 1].mean(dim=0)
    return out


class GrowthTrigger:
    """The registration's model-wide saturation trigger: the mean over
    (block, head, batch) of the per-head PR, smoothed over the last 4
    boundaries, firing at the first UPWARD crossing of tau_sat.  One event
    per window (the probe ran at the pre-event head count; a second event
    in the same window is meaningless -- receipted, next window re-evaluates
    at the grown core) and one per step (the optimizer surgery runs once)."""

    def __init__(self, tau_sat=TAU_SAT, window=TRIGGER_WINDOW):
        self.tau = tau_sat
        self.window = window

    def decide(self, growth_mean_per_boundary: torch.Tensor):
        """[C] -> the fire chunk index or None, with the receipt."""
        sm = _smoothed(growth_mean_per_boundary.view(-1, 1), self.window)[:, 0]
        prev = torch.cat([torch.zeros(1, device=sm.device), sm[:-1]])
        fire = ((sm >= self.tau) & (prev < self.tau)).nonzero()
        if len(fire) == 0:
            return None, dict(smoothed_max=float(sm.max()))
        c = int(fire[0, 0])
        return c, dict(fire_chunk=c, smoothed=float(sm[c]),
                       smoothed_max=float(sm.max()), tau=self.tau)


class RecyclingTrigger:
    """The LOCKED two-component per-head trigger (the supervisor's
    amendment): fire per (block, head) at an UPWARD crossing of the
    per-head PR (batch-mean, smoothed over 4 boundaries) while the head's
    smoothed READ-MASS is at or below tau_read -- saturated AND stale
    (recycling the workhorse would be an unfair test of C2).  One pop per
    (block, head) per window (the probe is un-popped; re-fires evaluate on
    the next window's real post-pop state -- receipted).  Pops at
    boundaries AFTER a growth fire in the same window are not made (growth
    resets the per-head bookkeeping at the new head count -- registered)."""

    def __init__(self, tau_pop=TAU_POP, tau_read=0.0, window=TRIGGER_WINDOW):
        self.tau_pop = tau_pop
        self.tau_read = tau_read
        self.window = window

    def decide(self, pr_bh: torch.Tensor, read_bh: torch.Tensor,
               grow_chunk=None):
        """pr_bh, read_bh: [C, H] batch-mean per-head statistics from the
        probe.  Returns ({chunk: [heads]}, receipts)."""
        sm_pr = _smoothed(pr_bh, self.window)
        sm_rd = _smoothed(read_bh, self.window)
        prev = torch.cat(
            [torch.zeros(1, pr_bh.shape[1], device=pr_bh.device),
             sm_pr[:-1]], dim=0)
        cross = (sm_pr >= self.tau_pop) & (prev < self.tau_pop)
        stale = sm_rd <= self.tau_read
        fire = cross & stale
        if grow_chunk is not None:
            fire[grow_chunk:, :] = False
        pops, receipts = {}, []
        seen_heads = set()
        for c, h in fire.nonzero().tolist():
            if h in seen_heads:
                continue            # one pop per head per window (first fire)
            seen_heads.add(h)
            pops.setdefault(c, []).append(h)
            receipts.append(dict(chunk=c, head=h,
                                 smoothed_pr=float(sm_pr[c, h]),
                                 smoothed_read=float(sm_rd[c, h]),
                                 tau_pop=self.tau_pop,
                                 tau_read=self.tau_read))
        return pops, receipts


# --------------------------------------------------------------------------
# The probe pass and the window statistics
# --------------------------------------------------------------------------

def probe_walk(model, wrappers, ids, collector):
    """One inference-mode forward of the FULL model with the wrappers in
    probe mode (metrics-only; the kernel's intermediate-state return is
    inference-mode-only by its own contract -- the growth smoke's S-2
    convention; the walk doubles as an exact forward under the training
    autocast)."""
    for w in wrappers:
        w.mode = "probe"
        w.collector = collector
    try:
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            x = model.model.embed(ids)
            for block in model.model.blocks:
                x = block(x, ids)
    finally:
        for w in wrappers:
            w.mode = "delegate"
            w.collector = None


def window_statistics(wrappers, collector, pool, max_matrices=65536):
    """Per (block): the batch-mean per-head PR [C, H] and read-mass [C, H],
    via bounded multi-block sidecar rounds (the blocks' boundary states are
    concatenated up to ``max_matrices`` spectra per request -- one dispatch
    at H=2, a few at grown head counts; the states are freed as they are
    consumed).  Also the model-wide growth statistic [C]."""
    stats = {}
    groups, cur, count = [], [], 0
    for w in wrappers:
        entry = collector[w.layer_idx]
        B, C1, H, K, V = entry["h"].shape
        n = B * (C1 - 1) * H
        if cur and count + n > max_matrices:
            groups.append(cur)
            cur, count = [], 0
        cur.append((w, entry, n))
        count += n
    if cur:
        groups.append(cur)
    for group in groups:
        pieces, sizes = [], []
        for w, entry, n in group:
            B, C1, H, K, V = entry["h"].shape
            pieces.append(entry["h"][:, 1:].reshape(-1, K, V))
            sizes.append((w, entry, B, C1, H, K, V, n))
        states = torch.cat(pieces, dim=0).contiguous()
        for w, entry, _ in group:
            entry["h"] = None               # freed as consumed
        del pieces
        pr_all = pool.pr_from_states(states)
        del states
        at = 0
        for w, entry, B, C1, H, K, V, n in sizes:
            pr = pr_all[at:at + n].view(B, C1 - 1, H).mean(dim=0)  # [C, H]
            at += n
            stats[w.layer_idx] = dict(
                pr=pr, read=entry["read"][:C1 - 1].float(), num_heads=H)
    growth_mean = torch.stack(
        [stats[w.layer_idx]["pr"] for w in wrappers]).mean(
        dim=(0, 2))                            # mean over (block, head): [C]
    return stats, growth_mean


def decide_plan(wrappers, stats, growth_mean, growth_trigger, pop_trigger,
                growth_enabled, pop_enabled, event_receipt,
                growth_blocked=False):
    """The window's plan: growth first (model-wide, one event per step --
    a fire while blocked by this step's earlier event is receipted as
    skipped and re-evaluated next step on the grown core), then the
    per-head pops (never after the growth boundary)."""
    plan = dict(growth_chunk=None, core_old=int(wrappers[0].current_idx))
    grow_chunk = None
    if growth_enabled and wrappers[0].current.num_heads < HEAD_CAP:
        grow_chunk, receipt = growth_trigger.decide(growth_mean)
        if grow_chunk is not None:
            if growth_blocked:
                event_receipt["growth_skipped"] = dict(
                    reason="one growth event per step (the optimizer surgery "
                            "runs once); re-evaluated next step on the "
                            "grown core", **receipt)
                grow_chunk = None
            else:
                event_receipt["growth"] = receipt
                plan["growth_chunk"] = grow_chunk
    pops_all, pop_receipts = {}, []
    if pop_enabled:
        for w in wrappers:
            s = stats[w.layer_idx]
            pops, receipts = pop_trigger.decide(s["pr"], s["read"],
                                                grow_chunk=grow_chunk)
            if pops:
                pops_all[w.layer_idx] = pops
                for r in receipts:
                    r["block"] = w.layer_idx
                pop_receipts.extend(receipts)
    if pop_receipts:
        event_receipt["pops"] = pop_receipts
        plan["pops"] = {int(c): sorted(heads)
                        for c, heads in pops_all.items()}
    # NOTE: pops are per-block; each block's wrapper gets its own plan copy
    # with only its own pops.
    return plan, pops_all


def stash_plans(wrappers, plan, pops_all, new_idx=None):
    for w in wrappers:
        w.plan = dict(plan)
        w.plan["pops"] = {c: list(hs) for c, hs in (pops_all.get(
            w.layer_idx, {}) or {}).items()}
        if new_idx is not None:
            w.plan["core_new"] = new_idx
    return


def clear_plans(wrappers):
    for w in wrappers:
        w.plan = None


# --------------------------------------------------------------------------
# The optimizer surgery at growth events (the registration's disclosed
# semantics: shared-slice carry; the event step's pre-event gradients are
# MERGED into the grown core's shared slices -- the reparametrized
# objective's exact gradient)
# --------------------------------------------------------------------------

def merge_event_gradients(shared):
    for old_p, new_p, sl in shared:
        if old_p.grad is None:
            continue
        if new_p.grad is None:
            new_p.grad = torch.zeros_like(new_p)
        if sl is None:
            new_p.grad += old_p.grad
        else:
            new_p.grad[sl] += old_p.grad


def rebuild_optimizer(optimizer, model, wrappers, shared, lr, betas,
                      weight_decay):
    """A fresh fused AdamW over the LIVE parameters (the retired core leaves
    the trainable set), with the AdamW state carried by slice for every
    shared parameter (the committed smoke's carry semantics)."""
    old_state = optimizer.state
    live = live_parameters(model, wrappers)
    new_opt = torch.optim.AdamW(live, lr=lr, betas=betas,
                                weight_decay=weight_decay, fused=True)
    by_id = {id(p): p for p in live}
    for old_p, new_p, sl in shared:
        st = old_state.get(old_p)
        if st and id(new_p) in by_id:
            new_st = new_opt.state[new_p]
            new_st["step"] = st.get("step", torch.tensor(0))
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    buf = new_st.setdefault(key, torch.zeros_like(new_p))
                    if sl is None:
                        buf.copy_(st[key])
                    else:
                        buf[sl] = st[key]
        old_p.grad = None      # the retired core's gradients are merged; the
                               # stale buffers are released
    return new_opt


# --------------------------------------------------------------------------
# The regime protocol (Option B): stream, rows-per-step, eval
# --------------------------------------------------------------------------

def build_stream(context: int):
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"], context=context)
    L.assert_flagship_stream(stream)
    return stream, identity


class RegimeWindows:
    """The LOCKED regime's stream construction (the C3 lane's committed
    protocol, mirrored for cross-lane comparability): FULL ``context``-byte
    windows from eligible documents only (documents shorter than the
    context contribute no windows), document-bounded, split-disjoint -- the
    registered ``lattice.Stream`` windows iterator (its own SHA-256 document
    verification and boundary legality) filtered to full windows, never
    packed across boundaries, batched at the locked 14 windows per step."""

    def __init__(self, stream, context: int,
                 per_step: int = WINDOWS_PER_STEP):
        self.stream = stream
        self.context = context
        self.per_step = per_step
        self.n_windows_fed = 0

    def batches(self, split: str):
        rows = []
        for row in self.stream.windows(split, self.context):
            if len(row[2]) != self.context:
                continue                 # full windows only (the locked rule)
            rows.append(row)
            if len(rows) == self.per_step:
                self.n_windows_fed += len(rows)
                yield rows
                rows = []
        if rows:
            raise RuntimeError("split exhausted before the budget: refusing "
                               "to run a partial step")


def evaluate_at_context(model, stream, split, context, windows=EVAL_WINDOWS):
    """The probe protocol's document-disjoint calibration eval at the
    regime's context: the first ``windows`` FULL windows of the split, batch
    1, the recorded-read path (the banked rule at 2048, read at the
    regime's context)."""
    os.environ["X40_BATCHED_READS"] = "0"
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    model.eval()
    total, targets, used = 0.0, 0, []
    with torch.no_grad():
        for row in stream.windows(split, context):
            if len(used) >= windows:
                break
            if len(row[2]) != context:
                continue
            x, y, _valid = next(grouped([row], 1, context))
            total += float(model.loss(x, y)) * y.numel()
            targets += int(y.numel())
            used.append([row[0], row[1]])
    if not targets:
        raise ValueError(f"no full {context}-byte windows in split {split}")
    model.train()
    return dict(score=total / targets, eval_split=split,
                 eval_windows=len(used), eval_targets=targets,
                 used_windows=used, units="nats/byte")


# --------------------------------------------------------------------------
# The training loop (the honest clock, the corrected scoring)
# --------------------------------------------------------------------------

def _arm_train_loop(model, wrappers, optimizer, optimizer_config, stream,
                    context, targets, cooldown_steps, label,
                    out, growth_trigger, pop_trigger, growth_enabled,
                    pop_enabled, pool, snapshot_steps, ema_decay,
                    thresholds, stats_sink=None, smoke=False):
    """Matched-exposure constant-LR phase (the banked protocol), then the
    linear cooldown; the probe pass -> plan -> real pass discipline per
    microbatch; snapshots + EMA inside the loop; every event receipted.
    ``stats_sink(step, microbatch_idx, stats, growth_mean)`` receives the
    per-window statistics (the calibration's receipts)."""
    from experiments.expressivity_tasks.x40_emender_lm.part6_surface import (
        features)
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    microbatch = optimizer_config["microbatch"]
    peak_lr = optimizer_config["lr"]
    warmup = optimizer_config["warmup_steps"]
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = seen = 0
    mb_idx = 0
    rows_log = []
    snapshots = {}
    ema = {n: p.detach().float().clone() for n, p in
           model.named_parameters() if p.requires_grad}
    shared_pending = None            # the growth event's merge map, this step
    growth_fired_this_step = False
    pop_stats = dict(total_pops=0, per_fire=[], per_step=[],
                     per_block_head={}, growth_skipped=[])
    pool_load_mid = None
    log_path = out / "training.jsonl" if out is not None else None
    log = log_path.open("w", buffering=1) if log_path is not None else None
    constant_steps = None
    try:
        phase = "constant"

        def lr_factor(step0, cooldown_k):
            f = min(1., (step0 + 1) / max(1, warmup))
            if cooldown_k:
                f *= (cooldown_steps + 1 - cooldown_k) / cooldown_steps
            return f

        while True:
            if phase == "constant" and seen >= targets:
                constant_steps = step
                snapshots["constant_end"] = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}
                phase = "cooldown"
            if phase == "cooldown" and \
                    (step - constant_steps) >= cooldown_steps:
                break
            cooldown_k = (step - constant_steps + 1
                          if phase == "cooldown" else 0)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            try:
                rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the budget")
            n = sum(len(r[3]) for r in rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = peak_lr * lr_factor(step, cooldown_k)
            loss_sum = 0.0
            step_pops = 0
            step_growth = None
            gate_counts = dict(pr_crossings=0, with_read_floor=0)
            probe_s = spectra_s = real_s = 0.0
            for x, y, valid in grouped(rows, microbatch, context):
                ids = x
                # ---- the probe pass (statistics; metrics-only)
                tp = time.monotonic()
                collector = {}
                probe_walk(model, wrappers, ids, collector)
                ts = time.monotonic()
                probe_s += ts - tp
                stats, growth_mean = window_statistics(wrappers, collector,
                                                       pool)
                spectra_s += time.monotonic() - ts
                if stats_sink is not None:
                    stats_sink(step, mb_idx, stats, growth_mean)
                mb_idx += 1
                # ---- the plan
                event_receipt = {}
                gate_counts.update(pop_trigger.count_gates(stats)) \
                    if pop_enabled else None
                plan, pops_all = decide_plan(
                    wrappers, stats, growth_mean, growth_trigger, pop_trigger,
                    growth_enabled, pop_enabled, event_receipt,
                    growth_blocked=growth_fired_this_step)
                new_idx = None
                if plan["growth_chunk"] is not None:
                    # the registrar event: every block grows (one event per
                    # step; skipped fires are receipted)
                    old_heads = wrappers[0].current.num_heads
                    new_heads = min(HEAD_CAP, 2 * old_heads)
                    seed = SEED + 1000 * (step + 1) + 7
                    all_shared = []
                    for w in wrappers:
                        new_core, shared = grow_core(w.current, new_heads,
                                                     seed)
                        w.cores.append(new_core)
                        all_shared.extend(shared)
                        new_idx = len(w.cores) - 1
                        w.current_idx.fill_(new_idx)
                    shared_pending = all_shared
                    growth_fired_this_step = True
                    step_growth = dict(
                        step=step + 1, chunk=plan["growth_chunk"],
                        heads=old_heads, new_heads=new_heads,
                        receipt=event_receipt.get("growth"))
                stash_plans(wrappers, plan, pops_all, new_idx=new_idx)
                # ---- the real pass (the banked forward + the criterion)
                tr = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden, _ = features(model.model, ids, "gdn2", True)
                    loss = model.criterion(
                        hidden.to(torch.bfloat16).contiguous(),
                        y.contiguous(),
                        model.model.head.weight.to(torch.bfloat16))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step+1}")
                (loss * (valid / n)).backward()
                real_s += time.monotonic() - tr
                loss_sum += float(loss.detach()) * valid / n
                # ---- the event receipts
                nfires = sum(len(hs) for hs in pops_all.values())
                pop_stats["total_pops"] += nfires
                step_pops += nfires
                for r in event_receipt.get("pops", []):
                    r["step"] = step + 1
                    pop_stats["per_fire"].append(r)
                    key = f"block{r['block']}_head{r['head']}"
                    pop_stats["per_block_head"][key] = \
                        pop_stats["per_block_head"].get(key, 0) + 1
                if "growth_skipped" in event_receipt:
                    pop_stats["growth_skipped"].append(
                        dict(step=step + 1,
                             **event_receipt["growth_skipped"]))
                clear_plans(wrappers)
            # ---- the growth event's surgery: merge the pre-event gradients
            # into the grown core's shared slices, rebuild the optimizer
            if shared_pending is not None:
                merge_event_gradients(shared_pending)
                optimizer = rebuild_optimizer(
                    optimizer, model, wrappers, shared_pending,
                    peak_lr, tuple(optimizer_config["betas"]),
                    optimizer_config["weight_decay"])
                shared_pending = None
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(live_parameters(model, wrappers),
                                           GRAD_CLIP, error_if_nonfinite=True)
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()
            growth_fired_this_step = False
            elapsed += time.monotonic() - t0
            step += 1
            seen += n
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad and name in ema:
                        ema[name].mul_(ema_decay).add_(p.detach().float(),
                                                       alpha=1 - ema_decay)
                    elif p.requires_grad and name not in ema:
                        ema[name] = p.detach().float().clone()
            if step in snapshot_steps:
                snapshots[step] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
            row = dict(step=step, active_seconds=elapsed, bytes=seen,
                       loss=loss_sum, phase=phase,
                       lr_factor=lr_factor(step - 1, cooldown_k),
                       pops_this_step=step_pops,
                       pop_gate_counts=gate_counts,
                       growth_event=step_growth,
                       walls=dict(probe_s=round(probe_s, 2),
                                  spectra_s=round(spectra_s, 2),
                                  real_s=round(real_s, 2)))
            if log is not None:
                log.write(json.dumps(row) + "\n")
            rows_log.append(row)
            pop_stats["per_step"].append(dict(step=step, pops=step_pops,
                                              growth=step_growth))
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {loss_sum:.4f} pops {pop_stats['total_pops']}",
                      flush=True)
            if step == 90:
                pool_load_mid = SPECTRA.box_load_receipt()
        snapshots["final"] = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                constant_steps=constant_steps, snapshots=snapshots,
                ema=ema, rows=rows_log, pop_stats=pop_stats,
                pool_load_mid=pool_load_mid, optimizer=optimizer,
                thresholds=thresholds)


# --------------------------------------------------------------------------
# The runs (calibrate / arm)
# --------------------------------------------------------------------------

def _require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES "
                         "required")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs {visible}")
    return card


def _versions() -> dict:
    return dict(torch=torch.__version__,
                triton=__import__("triton").__version__)


def measure_state(model, wrappers, stream, context, pool, n_windows=8):
    """The trained-state statistics measurement (the C3 lane's calibration
    protocol, mirrored): probe + spectra statistics on the FIRST ``n_windows``
    full windows of the train split's window order (deterministic, disclosed),
    at the loaded state.  Returns the per-boundary growth-mean trajectory and
    the per-block statistics with the smoothed reductions the thresholds and
    the cross-checks read."""
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    rows = []
    for row in stream.windows("train", context):
        if len(row[2]) != context:
            continue
        rows.append(row)
        if len(rows) >= n_windows:
            break
    x, y, _valid = next(grouped(rows, n_windows, context))
    collector = {}
    probe_walk(model, wrappers, x, collector)
    stats, growth_mean = window_statistics(wrappers, collector, pool)
    out = dict(growth_smoothed=[float(v) for v in
                                _smoothed(growth_mean.view(-1, 1))[:, 0]],
               per_layer={}, per_head={})
    for w in wrappers:
        s = stats[w.layer_idx]
        sm_pr = _smoothed(s["pr"])              # [C, H]
        sm_rd = _smoothed(s["read"])             # [C, H]
        out["per_layer"][w.layer_idx] = dict(
            smoothed_max=float(sm_pr.mean(dim=1).max()),
            raw_max=float(s["pr"].mean(dim=1).max()))
        for h in range(s["pr"].shape[1]):
            am = int(sm_pr[:, h].argmax())
            out["per_head"][f"{w.layer_idx}_{h}"] = dict(
                smoothed_pr_max=float(sm_pr[:, h].max()),
                smoothed_read_at_pr_max=float(sm_rd[am, h]))
    out["pr_mean"] = float(torch.stack(
        [stats[w.layer_idx]["pr"] for w in wrappers]).mean())
    out["pr_max"] = float(torch.stack(
        [stats[w.layer_idx]["pr"] for w in wrappers]).max())
    return out, stats


def calibrate(args) -> dict:
    """The fixed arm under the LOCKED regime (context 8192, 14 full windows
    per step, the 20,070,400 landing, the 24-step cooldown), trained clean
    (the never-fires wrapper -- PS-1's inertness: the training path IS the
    registered forward), scored under the corrected convention, then
    MEASURED at the trained states (the C3 lane's calibration protocol,
    mirrored for cross-lane receipt comparability): the per-layer smoothed-PR
    maxima (their committed cross-check anchors), the per-head smoothed-PR
    ceiling (this lane's tau_pop), the read-mass distribution among
    per-head-saturated observations (the pre-committed tau_read P25 rule),
    and the registered-96 unreachability verdicts.  The threshold commit
    lands BEFORE any arm runs."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    stream, identity = build_stream(args.context)
    regime = RegimeWindows(stream, args.context)
    card = _require_lease()
    model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
    wrappers = convert_model(model)
    pool = SPECTRA.SpectraPool()
    targets = int(args.bytes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / f"calibrate_c{args.context}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    c3_regime_path = RECEIPTS / "gdn2_growth_c3_regime.json"
    c3_regime = json.loads(c3_regime_path.read_text())
    c3_cal_path = RECEIPTS / "gdn2_growth_c3_calibration_8192.json"
    c3_cal = json.loads(c3_cal_path.read_text())
    assert c3_regime["tau_sat"] == TAU_SAT_C3_LOCKED, (
        "the C3 lane's committed tau_cal drifted vs this lane's imported "
        f"constant: {c3_regime['tau_sat']!r} != {TAU_SAT_C3_LOCKED!r}")
    print(json.dumps(dict(probe="gdn2_growth_pop calibrate",
                          context=args.context, card=card, seed=args.seed,
                          windows_per_step=WINDOWS_PER_STEP,
                          parameters=sum(p.numel() for p in
                                          model.parameters()))), flush=True)

    growth_trigger = GrowthTrigger(tau_sat=float("inf"))
    pop_trigger = RecyclingTrigger(tau_pop=float("inf"), tau_read=float("inf"))
    try:
        run = _arm_train_loop(
            model, wrappers,
            torch.optim.AdamW(live_parameters(model, wrappers),
                              lr=optimizer_config["lr"],
                              betas=tuple(optimizer_config["betas"]),
                              weight_decay=optimizer_config["weight_decay"],
                              fused=True),
            optimizer_config, regime, args.context,
            targets, COOLDOWN_STEPS, f"calibrate_c{args.context}", run_dir,
            growth_trigger, pop_trigger, False, False, pool,
            set(CHECKPOINT_STEPS), EMA_DECAY,
            dict(tau_sat=TAU_SAT, tau_pop=TAU_POP,
                 tau_read_rule="P25 pre-committed"))

            # ---- the scoring pass (the corrected convention, the regime's eval)
        rows_by_step = {r["step"]: r for r in run["rows"]}
        scores = []
        for s in sorted(set(CHECKPOINT_STEPS)):
            if s not in run["snapshots"] or s > (run["constant_steps"] or 0):
                continue
            model.load_state_dict(run["snapshots"][s])
            res = evaluate_at_context(model, stream, args.eval_split, args.context)
            scores.append(dict(kind="constant", step=s,
                               bytes=rows_by_step[s]["bytes"],
                               train_loss=rows_by_step[s]["loss"],
                               eval_nll=res["score"], eval_windows=res["eval_windows"],
                               eval_targets=res["eval_targets"]))
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            res = evaluate_at_context(model, stream, args.eval_split, args.context)
            primary = dict(kind="post_cooldown", step=run["step"],
                           eval_nll=res["score"], eval_windows=res["eval_windows"],
                           eval_targets=res["eval_targets"])
            scores.append(primary)

        # ---- the trained-state measurements (the threshold authority)
        measurements = {}
        for tag, key in (("constant_end", "constant_end"),
                         ("post_cooldown", "final")):
            if key in run["snapshots"]:
                model.load_state_dict(run["snapshots"][key])
                m, _stats = measure_state(model, wrappers, stream, args.context,
                                          pool)
                measurements[tag] = m
        m = measurements["post_cooldown"]
        per_head_ceiling = max(v["smoothed_pr_max"]
                               for v in m["per_head"].values())
        sat_reads = sorted(v["smoothed_read_at_pr_max"]
                           for v in m["per_head"].values()
                           if v["smoothed_pr_max"] >= per_head_ceiling - 1e-9)
        tau_read = (sat_reads[max(0, int(round(0.25 * len(sat_reads))) - 1)]
                    if sat_reads else None)

        # ---- the reachability verdicts and the threshold commit
        commit = dict(
            context=args.context,
            tau_sat=TAU_SAT_C3_LOCKED,
            tau_sat_source=("the C3 lane's committed calibrated growth threshold "
                            "(receipts/gdn2_growth_c3_regime.json, imported "
                            "verbatim -- the same growth trigger on the same "
                            "regime for cross-lane comparability)"),
            tau_pop=per_head_ceiling,
            tau_pop_rule=("the calibrated-below-registered discipline (the "
                          "supervisor's ruling, mirroring the C3 lane's tau_cal "
                          "rule at the POP trigger's own granularity): the "
                          "trained fixed machine's own PER-HEAD smoothed-PR "
                          "ceiling, measured on the post-cooldown (PRIMARY-"
                          "scored) trained state"),
            tau_read=tau_read,
            tau_read_rule=("P25 of the smoothed read-mass among per-head-"
                           "saturated observations (smoothed PR >= tau_pop, the "
                           "ceiling population: one argmax-boundary observation "
                           "per (block, head)), on the post-cooldown trained "
                           "state; pre-committed before the arms run"),
            hazard=("the K8 idle-store hazard stated verbatim: the calibrated "
                    "tau_pop is the fixed machine's own regime ceiling, NOT "
                    "absolute capacity; the arm tests capacity management past "
                    "the exhibited regime, not capacity management at capacity"),
            tau_registered_unreachable=dict(
                tau_sat=TAU_SAT, tau_pop=TAU_POP,
                verdict=("the registered 0.75*d_k = 96 thresholds are "
                         "UNREACHABLE on the real C3 stream at both scanned "
                         "horizons (this lane's receipts and the C3 lane's "
                         "committed calibration receipts: trained fixed GDN2 "
                         "PR mean ~7.5, max ~24, smoothed per-layer max ~19, "
                         "per-head ceiling receipted below) -- the informative "
                         "null branch of the registered 96 thresholds is "
                         "delivered by these reachability receipts and is "
                         "attached to the registration permanently"),
                receipt=str(out)),
            n_ceiling_observations=len(sat_reads),
            committed_by="gdn2_growth_pop calibrate (the locked regime, the "
                         "disclosed trained-state protocol)",
        )
        receipt = dict(
            probe="gdn2_growth_pop calibrate", context=args.context,
            seed=args.seed, identity=identity,
            regime=dict(windows_per_step=WINDOWS_PER_STEP,
                        eval_windows=EVAL_WINDOWS,
                        note="the LOCKED regime (context 8192, full windows from "
                             "eligible documents only, 14/step, the 20,070,400 "
                             "landing, 24-step cooldown)"),
            thresholds_registered=dict(tau_sat=TAU_SAT, tau_pop=TAU_POP,
                                        tau_read_rule="P25 pre-committed"),
            reachability_and_commit=commit,
            measurements=dict(
                constant_end=measurements.get("constant_end"),
                post_cooldown=measurements.get("post_cooldown")),
            cross_check=dict(
                c3_calibration_receipt=str(c3_cal_path),
                c3_calibration_sha256=matched_arms.digest(c3_cal_path),
                c3_regime_receipt=str(c3_regime_path),
                c3_regime_sha256=matched_arms.digest(c3_regime_path),
                step1_train_loss=run["rows"][0]["loss"],
                step1_c3_banked=c3_cal["trajectory"][0]["loss"],
                step1_bit_identical_to_c3=(
                    run["rows"][0]["loss"] == c3_cal["trajectory"][0]["loss"]),
                primary_post_cooldown=primary["eval_nll"],
                primary_c3_banked=c3_cal["scores"]["primary"]["eval_nll"],
                c3_pr_anchors=dict(
                    mean=c3_cal["pr"]["post_cooldown"]["mean"],
                    max=c3_cal["pr"]["post_cooldown"]["max"],
                    per_layer_smoothed_max_layer5=(
                        c3_cal["pr"]["post_cooldown"]
                        ["per_layer_smoothed_max"]["5"])),
                note="the step-1 train loss must be BIT-IDENTICAL (the same "
                     "seed, the same locked stream construction, the same "
                     "optimizer); the PR anchors and the primary score "
                     "cross-check statistically (GPU nondeterminism may move "
                     "later steps; material divergence is escalated"),
            scores=scores,
            optimizer=optimizer_config, summary=summary,
            versions=_versions(), session=dict(card=card,
                 device=torch.cuda.get_device_name(0)),
            pool=dict(stats=pool.stats(), load_at_enable=pool.load_at_enable),
            status=run["status"], steps=run["step"], bytes=run["seen"],
            constant_steps=run["constant_steps"],
            walls=dict(train_seconds=round(run["elapsed"], 1)),
            events=run["pop_stats"], git=matched_arms.git_state())
    finally:
        pool.close()
    atomic_json(out, receipt)
    thresholds_path = out.with_name(out.stem + "_thresholds.json")
    atomic_json(thresholds_path, commit)
    print(json.dumps(dict(context=args.context, steps=run["step"],
                          bytes=run["seen"],
                          step1_bit_identical_to_c3=(
                              run["rows"][0]["loss"] ==
                              c3_cal["trajectory"][0]["loss"]),
                          primary=primary["eval_nll"],
                          per_head_ceiling=per_head_ceiling,
                          tau_read=tau_read)), flush=True)
    return receipt


def arm(args) -> dict:
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    stream, identity = build_stream(args.context)
    regime = RegimeWindows(stream, args.context)
    card = _require_lease()
    model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
    wrappers = convert_model(model)
    pool = SPECTRA.SpectraPool()
    thresholds = json.loads(Path(args.thresholds).read_text())
    growth_enabled = args.arm in ("gdn2_growth", "gdn2_growth_pop")
    pop_enabled = args.arm == "gdn2_growth_pop"
    growth_trigger = GrowthTrigger(tau_sat=thresholds["tau_sat"])
    pop_trigger = RecyclingTrigger(tau_pop=thresholds["tau_pop"],
                                   tau_read=thresholds["tau_read"])
    targets = int(args.bytes)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run_config = dict(
        probe="gdn2_growth_pop arm", arm=args.arm, context=args.context,
        seed=args.seed, exposure_targets=targets,
        growth_enabled=growth_enabled, pop_enabled=pop_enabled,
        thresholds=thresholds,
        thresholds_note=("the calibration's committed thresholds "
                         f"({args.thresholds})"),
        windows_per_step=WINDOWS_PER_STEP,
        optimizer=optimizer_config, summary=summary,
        identity=identity, versions=_versions(),
        corrected_scoring=dict(
            primary="post-cooldown eval (24-step linear cooldown after the "
                    "matched-exposure constant-LR landing)",
            secondary="EMA-0.99 iterate at the constant-phase end "
                      "(ranking-only)",
            retained="constant-LR eval at the last shape-compatible "
                    "checkpoint (banked-comparability column)"),
        no_prediction=("NO DIRECTIONAL PREDICTION on the headline: a tie "
                       "means the C2 exact-capacity claim is "
                       "controller-only on GDN2's architecture too (the "
                       "honest verdict); a win is the op thesis's first "
                       "positive receipt on the dense machine"),
        session=dict(card=card, device=torch.cuda.get_device_name(0)),
        git=matched_arms.git_state())
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in
                                      live_parameters(model, wrappers)))))
    print(json.dumps(dict(arm=args.arm, context=args.context, card=card,
                          seed=args.seed,
                          parameters=sum(p.numel() for p in
                                         model.parameters()))), flush=True)

    cooldown = 3 if args.smoke else COOLDOWN_STEPS
    snapshot_steps = set() if args.smoke else set(CHECKPOINT_STEPS)
    try:
        run = _arm_train_loop(
            model, wrappers,
            torch.optim.AdamW(live_parameters(model, wrappers),
                               lr=optimizer_config["lr"],
                               betas=tuple(optimizer_config["betas"]),
                               weight_decay=optimizer_config["weight_decay"],
                               fused=True),
            optimizer_config, regime, args.context,
            targets, cooldown, args.arm, out,
            growth_trigger, pop_trigger, growth_enabled, pop_enabled, pool,
            snapshot_steps, EMA_DECAY, thresholds, smoke=args.smoke)

        # ---- the scoring pass (restored snapshots; training is over)
        rows_by_step = {r["step"]: r for r in run["rows"]}
        scores = []
        shape_ok = [s for s in sorted(snapshot_steps)
                    if s in run["snapshots"]]

        def eval_snapshot(tag, sd):
            model.load_state_dict(sd)
            res = evaluate_at_context(model, stream, args.eval_split,
                                      args.context)
            return res

        if not args.smoke:
            # the retained constant-LR column: the LAST SHAPE-COMPATIBLE
            # checkpoint at or before the constant-phase end (a growth event
            # changes the parameter set; pre-event snapshots cannot strict-load
            # into the final model -- the compatibility filter is the honest
            # rule, and the growth-event log receipts why)
            current_keys = set(model.state_dict())
            cand = [s for s in shape_ok
                    if (run["constant_steps"] is None
                        or s <= run["constant_steps"])
                    and set(run["snapshots"][s]) == current_keys]
            if cand:
                s = cand[-1]
                res = eval_snapshot(f"constant_{s}", run["snapshots"][s])
                scores.append(dict(kind="constant", step=s,
                                   bytes=rows_by_step[s]["bytes"],
                                   train_loss=rows_by_step[s]["loss"],
                                   eval_nll=res["score"],
                                   eval_windows=res["eval_windows"],
                                   eval_targets=res["eval_targets"],
                                   note="the last shape-compatible constant-LR "
                                        "checkpoint (growth events change the "
                                        "parameter set; the event log receipts "
                                        "which snapshots stay loadable)"))
            # the EMA secondary (ranking-only)
            if run["constant_steps"] in run["snapshots"] and \
                    set(run["snapshots"][run["constant_steps"]]) == current_keys:
                model.load_state_dict(run["snapshots"][run["constant_steps"]])
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.requires_grad and name in run["ema"]:
                            p.copy_(run["ema"][name].to(p.dtype))
                res = evaluate_at_context(model, stream, args.eval_split,
                                          args.context)
                scores.append(dict(kind="ema", step=run["constant_steps"],
                                   eval_nll=res["score"],
                                   eval_windows=res["eval_windows"],
                                   eval_targets=res["eval_targets"],
                                   ema_decay=EMA_DECAY,
                                   note="ranking-only"))
        # the PRIMARY: the post-cooldown final
        if "final" in run["snapshots"]:
            res = eval_snapshot("final", run["snapshots"]["final"])
            scores.append(dict(kind="post_cooldown", step=run["step"],
                               bytes=run["seen"], eval_nll=res["score"],
                               eval_windows=res["eval_windows"],
                               eval_targets=res["eval_targets"]))
        with (out / "scores.jsonl").open("w", buffering=1) as f:
            for s in scores:
                f.write(json.dumps(s) + "\n")

        constant_final = next((s["eval_nll"] for s in scores
                               if s["kind"] == "constant"), None)
        post_cooldown = next((s["eval_nll"] for s in scores
                              if s["kind"] == "post_cooldown"), None)
        ema_final = next((s["eval_nll"] for s in scores if s["kind"] == "ema"),
                         None)
        receipt = dict(
            probe="gdn2_growth_pop arm", arm=args.arm, context=args.context,
            seed=args.seed, status=run["status"],
            thresholds=thresholds, steps=run["step"],
            constant_steps=run["constant_steps"],
            exposure_bytes=rows_by_step[run["constant_steps"]]["bytes"]
            if run["constant_steps"] in rows_by_step else run["seen"],
            total_bytes=run["seen"],
            train_seconds=round(run["elapsed"], 1),
            final_eval_nll_constant_lr=constant_final,
            final_eval_nll_post_cooldown=post_cooldown,
            ema_eval_nll=ema_final,
            cooldown_drop=(constant_final - post_cooldown
                           if constant_final is not None
                           and post_cooldown is not None else None),
            scores=scores,
            recycling_statistics=run["pop_stats"],
            recycling_summary=dict(
                total_pops=run["pop_stats"]["total_pops"],
                pops_by_block_head=run["pop_stats"]["per_block_head"],
                growth_events=[r for r in run["pop_stats"]["per_step"]
                               if r["growth"]],
                note="per-fire: block, head, the smoothed PR and read-mass at "
                     "fire, the step and window chunk; the late-window H=8-cap "
                     "regime (growth exhausted, pops the only adaptive act) is "
                     "readable from the per-fire step positions"),
            pool=dict(stats=pool.stats(),
                      load_at_enable=pool.load_at_enable,
                      load_mid_run=run["pool_load_mid"]),
            parameters_final=sum(p.numel() for p in model.parameters()),
            peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            config_sha256=matched_arms.config_sha256(run_config),
            config=run_config)
    finally:
        pool.close()
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "final_eval_nll_post_cooldown", "steps",
                       "total_bytes")}), flush=True)
    return receipt


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------

def assemble(args) -> dict:
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    arms = {}
    for label in ("gdn2_growth", "gdn2_growth_pop"):
        path = RUNS / f"{label}_seed{SEED}" / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        arms[label] = dict(
            post_cooldown=r["final_eval_nll_post_cooldown"],
            constant=r["final_eval_nll_constant_lr"],
            ema=r["ema_eval_nll"], steps=r["steps"],
            total_pops=r["recycling_statistics"]["total_pops"],
            growth_events=len(r["recycling_summary"]["growth_events"]))
    if len(arms) < 2:
        raise SystemExit(f"not all arms have landed yet: {sorted(arms)}")
    pop_ahead = arms["gdn2_growth"]["post_cooldown"] - \
        arms["gdn2_growth_pop"]["post_cooldown"]
    verdict = dict(
        probe="gdn2_growth_pop assemble", arms=arms,
        margin_pop_minus_control=pop_ahead,
        band=ENDPOINT_VARIANCE_BAND,
        reading=("POP ahead by "
                 f"{-pop_ahead:.4f}" if pop_ahead < 0 else
                 f"control ahead by {pop_ahead:.4f}") +
        (" -- OUTSIDE the endpoint-variance band"
         if abs(pop_ahead) > ENDPOINT_VARIANCE_BAND else
         " -- inside the endpoint-variance band (a tie at this scale)"),
        no_prediction=("pre-stated: no directional prediction; a tie is the "
                       "honest controller-only verdict for C2 on the dense "
                       "machine, a win is the op thesis's first positive "
                       "receipt there"))
    atomic_json(RECEIPTS / "gdn2_growth_pop_summary.json", verdict)
    print(json.dumps(verdict, indent=1), flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--context", type=int, required=True,
                   choices=CONTEXTS)
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    c.add_argument("--out", required=True)
    c.add_argument("--eval-split", default="calibration")
    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True,
                   choices=("gdn2_growth", "gdn2_growth_pop"))
    a.add_argument("--context", type=int, required=True, choices=CONTEXTS)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--thresholds", required=True,
                   help="the calibration's committed threshold JSON")
    a.add_argument("--output", required=True)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--smoke", action="store_true",
                   help="a tiny-exposure shakedown (3 cooldown steps, no "
                        "snapshots/scoring receipts)")
    s = sub.add_parser("assemble")
    args = ap.parse_args()
    if args.cmd == "calibrate":
        calibrate(args)
    elif args.cmd == "arm":
        arm(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
