"""C3 EXECUTION -- GDN2+growth vs the growable machine in the saturation regime.

The authority chain this driver runs under:

* ``PREREGISTRATION_GDN2_GROWTH.md`` §7 (operator-signed this session) -- the
  four arms and the falsifiable prediction structure P-G1/P-G2/P-G3 with the
  refutation clauses;
* the supervisor's C3 regime ruling (2026-11-22), which unblocks §7's deferred
  regime definition: **Option B** -- the long-horizon variant of the
  registered Pile stream (same documents, longer windows via
  ``lattice.Stream(context=...)``; NO lattice.py edit), horizon chosen
  EMPIRICALLY by scanning BOTH 8192 and 16384 during the τ_sat calibration
  and locking the SHORTEST horizon at which the registered τ_sat = 0.75·d_k
  = 96 is reachable on the trained fixed machine; document-bounded windows
  only, full windows from eligible documents (byte_length ≥ context+1), the
  eligible-document count receipted; fresh kernel-shape receipts at the
  longer T (impossibility escalated, never patched); the standing probe
  budget (20M-target matched-exposure landing, constant-LR phase + the
  24-step linear cooldown of ``RESULT_PROTOCOL_CORRECTION.md`` as the
  PRIMARY score, EMA-0.99 at the constant-phase end as the ranking-only
  secondary, the constant-phase eval retained for comparability); and the
  emender arm FRESH in-regime at matched exposure (the banked K8 receipts do
  not transfer to the long-horizon regime);
* the spectra sidecar ruling: every per-boundary participation-ratio spectrum
  is computed by ``x45_spectra_sidecar`` VERBATIM (imported, never re-written
  -- the cross-lane consistency ruling): 32 hard-capped single-thread CPU
  workers, shared-memory handoff, exact LAPACK, the GPU-driven main process
  untouched.

Protocol constants (the standing probe rule transplanted to full-window long
context, receipted in the regime lock BEFORE any growth arm runs):

* windows/step = ceil(20,000,000 / (ctx·180)) (14 at 8192, 7 at 16384) -- the
  constant phase lands at 175 steps and 20,070,400 targets at BOTH candidate
  horizons (the banked protocol's 180-step/20,022,866 landing is the T=2048
  geometry's own special case of the same 20M rule);
* cooldown = 24 linear steps (13.7% of 175, inside the Hägele 10-15% band;
  the corrected-scoring form of ``protocol_correction.cooldown_factor``),
  same stream/data order, LR factor (25-k)/24;
* one optimizer family across all four arms (the K6/K8/side-by-side
  convention: plain AdamW at the registered gdn2 nomination's transferred
  settings, identical numbers on both machines -- the emender arm's own
  banked lane recipe; the GDN2 arms' optimizer surgery at growth events is
  the smoke-receipted AdamW surgery, and no Muon arm is run);
* seed 3401 for every arm (the K8 probe's own convention);
* the eval = the first 24 FULL context windows of the document-disjoint
  calibration split (the banked first-24-rows filter would yield ZERO full
  windows at the long horizons; the C3 rule scans for full windows, the
  count receipted).

The pre-stated falsifiable predictions (committed in the regime-lock receipt
before the growth arms run; margins fixed per the lane's standing
endpoint-variance band 0.075 nats/byte):

* **P-G1 (the binding receipt).** In-regime the fixed GDN2 arm's state
  saturates: the smoothed mean per-head participation ratio reaches τ_sat on
  the trained fixed machine (the calibration receipts this directly).  The
  empirical shadow: the fixed arm's saturation/interference statistics pin at
  the ceiling while the grown arms' do not.
* **P-G2 (growth is THE differentiator).** Supported iff `gdn2_growth` beats
  `gdn2_fixed` by MORE than 0.075 nats/byte on the post-cooldown PRIMARY
  score AND the trigger fired in-regime (growth events receipted).  ≈ (within
  the band) or never-fired are both the declared informative null (the K8 tie
  branch), reported as such.
* **P-G3 (slots earn their keep).** Supported iff `emender_growable` beats
  `gdn2_growth` by MORE than 0.075 nats/byte post-cooldown at matched
  exposure, with state sizes and parameter deltas as first-class columns.
  `gdn2_growth ≥ emender_growable` (within the band) refutes the slot story's
  necessity at this scale.
* The fixed-8 size control separates grown-large from born-large; no
  directional prediction is pre-stated for it (the registration states none).

GDN2's code is READ-ONLY throughout: the growth mechanism is the
wrapper/registrar pattern from the committed ``gdn2_growth_smoke.py``
(``GrowingGDN2Core`` wraps each block's ``GatedDeltaNet2`` core; the
no-event path calls the core's own forward verbatim).  No ``k10_*``/``k11_*``
/``k12_*`` source, no PREREGISTRATION, no ``lattice.py`` is edited.

Subcommands::

    kshapes    fresh kernel/shape receipts at T in {8192, 16384}, H in {2,4,8}
    calibrate  the τ_sat calibration scan on BOTH horizons + the regime lock
    arm        one arm (gdn2_fixed | gdn2_growth | gdn2_fixed_8 |
               emender_growable); --smoke = the machinery shakedown
    assemble   the four-arm table + the P-G1/P-G2/P-G3 adjudication
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NDM_PIN_TRITON_AUTOTUNE", "0")

import numpy as np
import torch
import torch.nn.functional as F

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender import x45_spectra_sidecar as SPECTRA
from experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_smoke import (
    _chunk_call, _kernel_inputs, _readout)
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction import (
    COOLDOWN_STEPS, EMA_DECAY, cooldown_factor)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "c3_runs"
GROWTH_REG = ROOT / "PREREGISTRATION_GDN2_GROWTH.md"
GROWTH_SMOKE_RECEIPT = RECEIPTS / "gdn2_growth_smoke.json"
SIDECAR_SELFTEST = RECEIPTS / "x45_spectra_sidecar_selftest.json"

HEAD_DIM = 128
HEAD_V_DIM = 256                 # head_dim * expansion 2.0 (the registered arm)
SEGMENT = 64                     # the statistic boundary = the kernel's chunk
HEAD_CAP = 8                     # the snap set's cap: {2, 4, 8}
TAU_SAT = 0.75 * HEAD_DIM        # 96, registered; the calibration is its authority
SMOOTH = 4                       # smoothed over the last 4 boundaries
EXPOSURE_TARGETS = 20_000_000
HORIZONS = (8192, 16384)
SEED = 3401
EVAL_WINDOWS = 24
DIAG_WINDOWS = 4
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160)
GRAD_CLIP = 1.0
ENDPOINT_BAND = 0.075            # the lane's standing endpoint-variance band

ARMS = ("gdn2_fixed", "gdn2_growth", "gdn2_fixed_8", "emender_growable")

PREDICTIONS = dict(
    P_G1="the fixed GDN2 arm's state saturates in-regime: the smoothed mean "
         "per-head participation ratio reaches tau_sat=96 on the trained "
         "fixed machine (receipted by the calibration scan itself)",
    P_G2="gdn2_growth beats gdn2_fixed by MORE than 0.075 nats/byte on the "
         "post-cooldown PRIMARY score AND the tau_sat trigger fired "
         "in-regime; a tie within the band, or a never-fired trigger, is "
         "the declared informative null (the K8 tie branch), reported as such",
    P_G3="emender_growable beats gdn2_growth by MORE than 0.075 nats/byte "
         "post-cooldown at matched exposure (state sizes and parameter "
         "deltas first-class); gdn2_growth >= emender_growable within the "
         "band refutes the slot story's necessity at this scale",
    fixed_8="the born-at-8 size control: separates grown-large from "
            "born-large; no directional prediction pre-stated (the "
            "registration states none for the control)",
    band_nats_per_byte=ENDPOINT_BAND,
)


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def windows_per_step(ctx: int) -> int:
    """The 20M matched-exposure landing in ~180 constant-LR steps: the
    window count per step scales inversely with the window length (14 at
    8192, 7 at 16384; the landing is 175 steps / 20,070,400 targets at both
    candidate horizons)."""
    return max(1, math.ceil(EXPOSURE_TARGETS / (ctx * 180)))


def microbatch_for(ctx: int, machine: str) -> int:
    """Memory knob only (grad accumulation is exact): GDN2 checkpoints its
    blocks and takes ~65K tokens per microbatch; the emender walk's banked
    per-scan footprint needs ~16K tokens per microbatch (the banked probe's
    own 8x2048)."""
    if machine == "emender":
        return {8192: 2, 16384: 1}[ctx]
    return {8192: 8, 16384: 4}[ctx]


def require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("no GPU lease: run through the leased driver")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: held {card} vs visible {visible}")
    return card


# ------------------------------------------------------------------ the stream


def c3_stream(ctx: int):
    """The supervisor-ruled long-horizon variant of the registered flagship
    stream: the SAME verified documents, longer document-bounded windows
    (``lattice.Stream``'s own context parameter; lattice.py is read-only)."""
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"], context=ctx)
    L.assert_flagship_stream(stream)
    return identity, stream


def eligible_documents(stream, ctx: int) -> dict:
    """The eligibility receipt: per split, documents long enough to yield at
    least one full ctx window (byte_length >= ctx+1), and the full-window
    count they yield."""
    per_split = {}
    for split in ("train", "calibration", "final_test"):
        docs = windows = 0
        for d in stream.documents(split):
            docs += 1
            windows += (d["length"] - 1) // ctx if d["length"] > ctx else 0
        per_split[split] = dict(eligible_documents=docs,
                                full_windows_produced=windows)
    return per_split


def full_windows(stream, split: str, ctx: int, limit: int):
    """The C3 regime's windows: FULL ctx-byte windows only (eligible
    documents; tails excluded -- the regime definition, disclosed in the
    regime lock)."""
    out = []
    for row in stream.windows(split):
        if len(row[2]) == ctx:
            out.append(row)
            if len(out) >= limit:
                break
    if len(out) < limit:
        raise ValueError(f"split {split} has only {len(out)} full {ctx}-byte "
                         f"windows; the regime needs {limit}")
    return out


def train_batches(stream, ctx: int, batch: int):
    def generate():
        rows = []
        for row in stream.windows("train"):
            if len(row[2]) != ctx:
                continue                      # full windows only (the regime)
            rows.append(row)
            if len(rows) == batch:
                yield rows
                rows = []
        if rows:
            yield rows
    return generate()


def evaluate_ctx(model, windows, ctx: int) -> dict:
    """The banked eval rule mirrored at the regime's context: model.loss over
    FULL windows, batch 1 (no padding, no partial windows).  The matrix
    machine's recorded-read eval path is enforced exactly as the banked
    evals do (X40_BATCHED_READS=0); GDN2 ignores the knob."""
    os.environ["X40_BATCHED_READS"] = "0"
    was_training = model.training
    model.eval()
    total = targets = 0.0
    used = []
    with torch.no_grad():
        for row in windows:
            x, y, _ = next(grouped([row], 1, context=ctx))
            loss = model.loss(x, y)
            total += float(loss) * y.numel()
            targets += int(y.numel())
            used.append([row[0], row[1]])
    if was_training:
        model.train()
    return dict(score=total / targets, nll_heldout=total / targets,
                eval_split="calibration", eval_windows=len(used),
                eval_targets=int(targets), units="nats/byte",
                eval_context=ctx)


# ------------------------------------------------- PR / trigger / co-diagnostic


def pr_from_trajectory(pool, h: torch.Tensor) -> torch.Tensor:
    """Per-boundary participation ratio from the kernel's own trajectory:
    ``h[:, c+1]`` is the state AFTER chunk c ([B, C, H, 128, 256]); the
    empty initial state h[:, 0] is excluded; zero states guarded to PR=0
    (the sidecar's own convention)."""
    B, C, H, K, V = h.shape
    states = h[:, 1:].reshape(-1, K, V).contiguous()
    pr = pool.pr_from_states(states)
    return pr.view(B, C - 1, H)


def smoothed_series(mean_pr: np.ndarray, smooth: int = SMOOTH) -> np.ndarray:
    """The registered aggregation: mean over (batch, head) at the boundary,
    smoothed over the last ``smooth`` boundaries (partial means at the start
    of a span; reset on fire -- each span's series is computed fresh)."""
    out = np.empty_like(mean_pr)
    for c in range(len(mean_pr)):
        lo = max(0, c - smooth + 1)
        out[c] = mean_pr[lo:c + 1].mean()
    return out


def iota_from_trajectory(h: torch.Tensor, b: torch.Tensor, khat: torch.Tensor):
    """The non-gating co-diagnostic (registration §5), per (batch, head,
    segment): iota = mean_t ||S_entry^T (b_t . k_hat_t)|| / (||S_entry||_F
    ||b_t . k_hat_t||).  The trajectory carries num_chunks states where
    h[:, c] is chunk c's ENTRY state (h[:, 0] the empty initial state) --
    every trajectory entry is some segment's entry, exactly the smoke's
    own S-5 loop."""
    B, C, H, K, V = h.shape          # C = num_chunks
    T = b.shape[1]
    seg = T // C
    write = (b.float() * khat.float()).reshape(B, C, seg, H, K)
    s_entry = h.float()              # every entry is a segment entry state
    num = torch.linalg.norm(
        torch.einsum("bcshk,bchkv->bcshv", write, s_entry),
        dim=-1)                                               # [B,C,seg,H]
    den = (torch.linalg.norm(s_entry, dim=(-2, -1))[:, :, None]
           * torch.linalg.norm(write, dim=-1)).clamp(min=1e-12)
    return (num / den).mean(dim=2)                                  # [B, C, H]


# ------------------------------------------------------- the growth registrar

# core-level slice rules (the smoke's layer-level rules transposed to the
# core; parameter NAMES are stable across events because the wrapper keeps
# its ``core`` slot, so the optimizer/EMA carry is by name)
KEY_ROWS = ("q_proj.weight", "k_proj.weight", "b_proj.weight", "f_proj.1.weight")
VAL_ROWS = ("v_proj.weight", "w_proj.weight", "g_proj.1.weight", "g_proj.1.bias")
CONV_KEY = ("q_conv1d.weight", "k_conv1d.weight")
CONV_VAL = ("v_conv1d.weight",)
O_PROJ = "o_proj.weight"          # new INPUT columns (dim 1) zeroed at birth


def _prefix_slice(old_shape, new_shape):
    return tuple(slice(0, o) if o != n else slice(None)
                 for o, n in zip(old_shape, new_shape))


def grow_core(old_core, new_heads: int, seed: int, wake_up: bool = True):
    """The registrar (§3's birth protocol, core level): a fresh core at H'
    born at GDN2's OWN init recipes, the old core's parameters copied into
    the shared slices, and -- the single deviation -- o_proj's new input
    columns zeroed (event-time output invariance + the one-step wake-up)."""
    old_heads = old_core.num_heads
    old_key, old_val = old_heads * HEAD_DIM, old_heads * HEAD_V_DIM
    torch.manual_seed(seed)
    cls = type(old_core)
    new_core = cls(hidden_size=old_core.hidden_size, expand_v=old_core.expand_v,
                   head_dim=old_core.head_dim, num_heads=new_heads,
                   mode="chunk", use_short_conv=old_core.use_short_conv,
                   conv_size=old_core.conv_size,
                   allow_neg_eigval=old_core.allow_neg_eigval,
                   layer_idx=old_core.layer_idx)
    new_core = new_core.to(device=old_core.A_log.device,
                           dtype=old_core.A_log.dtype)
    param_map = []      # (name, old_param, new_param, prefix_slice)
    with torch.no_grad():
        for (name_old, p_old), (name_new, p_new) in zip(
                old_core.named_parameters(), new_core.named_parameters()):
            assert name_old == name_new, (name_old, name_new)
            if name_new == O_PROJ:
                p_new[:, :old_val] = p_old
                if wake_up:
                    p_new[:, old_val:] = 0.0
            elif name_new in KEY_ROWS or name_new in CONV_KEY:
                p_new[:old_key] = p_old
            elif name_new in VAL_ROWS or name_new in CONV_VAL:
                p_new[:old_val] = p_old
            elif name_new == "A_log":
                p_new[:old_heads] = p_old
            elif name_new == "dt_bias":
                p_new[:old_key] = p_old
            else:
                # head-count-independent (f_proj.0, g_proj.0, o_norm): whole
                assert p_old.shape == p_new.shape, name_new
                p_new.copy_(p_old)
            if p_old.shape != p_new.shape:
                param_map.append((name_new, p_old, p_new,
                                  _prefix_slice(p_old.shape, p_new.shape)))
    return new_core, param_map


class GrowthEvent:
    def __init__(self, layer_idx, boundary, old_core, new_core, param_map,
                 fire_pr, seed, name_prefix=""):
        self.layer_idx = layer_idx
        self.boundary = boundary
        self.heads_old = old_core.num_heads
        self.heads_new = new_core.num_heads
        self.old_core = old_core
        self.new_core = new_core
        # the param_map's names are MODEL-LEVEL (the wrapper's stable core
        # slot prefix + the core-relative name) so the optimizer/EMA state
        # maps -- keyed by model-level names -- never collide across layers
        self.param_map = [(name_prefix + n, p_old, p_new, sl)
                          for (n, p_old, p_new, sl) in param_map]
        self.fire_pr = float(fire_pr)
        self.seed = seed
        self.name_prefix = name_prefix
        with torch.no_grad():
            # birth values for the EMA initialization of the new regions
            self.birth = {name_prefix + n: p.detach().clone()
                          for n, p in new_core.named_parameters()}


class GrowingGDN2Core(torch.nn.Module):
    """The wrapper (GDN2's code read-only): the no-event path calls the
    core's OWN forward verbatim (bit-identical to the unwrapped machine by
    construction); the growth path is §3's probe/real two-pass at 64-token
    statistic boundaries -- the probe (one inference-mode kernel call with
    return_intermediate_states=True per span) plans the events and performs
    the registrar swaps; the real pass replays the window with autograd, one
    kernel call per (module, span), the carried state differentiable (the
    kernel's own dh0), pre-event spans on the OLD core (the registration's
    letter; orphaned gradients merged into the live core's shared slices at
    the step's optimizer boundary)."""

    def __init__(self, core, layer_idx, sink, cap=HEAD_CAP, tau=TAU_SAT,
                 grow=True, seed_base=SEED):
        super().__init__()
        self.core = core
        self.layer_idx = layer_idx
        self.sink = sink            # the driver's shared sink dict
        self.cap = cap
        self.tau = tau
        self.grow = grow
        self.seed_base = seed_base
        self.name_prefix = ""      # set by install_growth (model-level names)
        self.n_events = 0
        self.last_probe = None      # per-scan telemetry (the driver samples)
        self._pending_fire = None   # the fire awaiting its post-event receipt

    @property
    def num_heads(self):
        return self.core.num_heads

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=None, output_attentions=None, **kwargs):
        T = hidden_states.shape[1]
        if (not self.grow or not self.training or self.core.num_heads >= self.cap
                or use_cache or T < 2 * SEGMENT
                or past_key_values is not None):
            return self.core(hidden_states, attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             use_cache=use_cache, **kwargs)
        plan = self._probe(hidden_states, self.sink["pool"])
        # the probe ran under inference_mode INSIDE the ambient autocast
        # region, so autocast's weight-cast cache now holds INFERENCE-mode
        # bf16 casts of the projections; the grad-tracked pass below must
        # not reuse them (and the registrar swapped modules besides) -- the
        # documented pattern for parameter mutation under autocast
        torch.clear_autocast_cache()
        if plan is None:
            return self.core(hidden_states, attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             use_cache=use_cache, **kwargs)
        return self._replay(hidden_states, plan)

    # ------------------------------------------------------------- the probe

    def _probe(self, x, pool):
        T = x.shape[1]
        spans = [(0, T, self.core)]
        events = []
        t0, cur, carry = 0, self.core, None
        telemetry = dict(layer=self.layer_idx, heads=cur.num_heads,
                         spans=[], fires=[])
        while True:
            xs = x[:, t0:]
            with torch.inference_mode():
                q, k, v, g, b, w = _kernel_inputs(cur, xs)
                o, s, h = _chunk_call(cur, q, k, v, g, b, w,
                                      initial_state=carry,
                                      output_final_state=True,
                                      return_intermediate_states=True)
                pr = pr_from_trajectory(pool, h).cpu()          # [B,C-1,H]
                mean_series = pr.mean(dim=(0, 2)).numpy()
                smoothed = smoothed_series(mean_series)
                khat = F.normalize(
                    k.reshape(-1, cur.num_heads, HEAD_DIM).float(),
                    dim=-1).reshape(k.shape)
                iota = iota_from_trajectory(h, b, khat).mean(dim=(0, 2))
                fire_idx = int(np.argmax(smoothed >= self.tau)) \
                    if bool((smoothed >= self.tau).any()) else None
                if self._pending_fire is not None:
                    # the post-event per-head PR receipt: the first boundary
                    # AFTER the event (the new heads entered exact zeros)
                    self._pending_fire[
                        "post_event_first_boundary_pr_per_head"] = \
                        pr[:, 0, :].mean(0).tolist()
                    self._pending_fire["pr_mean_drop_at_event"] = \
                        self._pending_fire["fire_boundary_pr_mean"] \
                        - float(mean_series[0])
                    self._pending_fire = None
            telemetry["spans"].append(dict(
                t0=t0, heads=cur.num_heads,
                pr_mean=float(mean_series.mean()),
                pr_smoothed_max=float(smoothed.max()),
                pr_first=float(mean_series[0]), pr_last=float(mean_series[-1]),
                iota_mean=float(iota.mean()), iota_max=float(iota.max()),
                n_boundaries=int(len(mean_series))))
            if fire_idx is not None and cur.num_heads < self.cap:
                boundary = t0 + SEGMENT * (fire_idx + 1)
                if boundary <= T - SEGMENT:
                    new_heads = min(self.cap, 2 * cur.num_heads)
                    seed = (self.seed_base + 10007 * self.layer_idx
                            + 101 * self.n_events + new_heads)
                    new_core, param_map = grow_core(cur, new_heads, seed)
                    ev = GrowthEvent(self.layer_idx, boundary, cur, new_core,
                                     param_map, smoothed[fire_idx], seed,
                                     name_prefix=self.name_prefix)
                    events.append(ev)
                    fire_record = dict(
                        boundary=boundary, heads=cur.num_heads,
                        new_heads=new_heads,
                        fire_smoothed_pr=float(smoothed[fire_idx]),
                        fire_boundary_pr_mean=float(mean_series[fire_idx]),
                        fire_boundary_pr_per_head=
                            pr[:, fire_idx, :].mean(0).tolist(),
                        seed=seed)
                    telemetry["fires"].append(fire_record)
                    self._pending_fire = fire_record
                    spans[-1] = (t0, boundary, cur)
                    with torch.inference_mode():
                        # the kernel asserts a float32 initial state; the
                        # trajectory is autocast dtype, so upcast at the pad
                        padded = F.pad(
                            h[:, fire_idx + 1].float(),
                            (0, 0, 0, 0, 0, new_heads - cur.num_heads))
                    t0, cur, carry = boundary, new_core, padded
                    spans.append((boundary, T, cur))
                    self.n_events += 1
                    continue
            break
        self.last_probe = telemetry
        if not events:
            return None
        self.core = spans[-1][2]        # the live core is the grown one
        self.sink["events"].extend(events)
        return spans

    # ------------------------------------------------------------- the replay

    def _replay(self, x, spans):
        outs = []
        carry = None
        for i, (t0, t1, core) in enumerate(spans):
            xs = x[:, t0:t1]
            q, k, v, g, b, w = _kernel_inputs(core, xs)
            last = i == len(spans) - 1
            o, s, _ = _chunk_call(core, q, k, v, g, b, w,
                                  initial_state=carry,
                                  output_final_state=not last)
            if not last:
                nxt = spans[i + 1][2]
                # the zero-injection: the new heads' state enters as exact
                # zeros at the event (the registration §3); the kernel
                # asserts a float32 initial state, so the carry is upcast
                carry = F.pad(s.float(),
                              (0, 0, 0, 0, 0, nxt.num_heads - core.num_heads))
            outs.append(_readout(core, o, xs))
        return torch.cat(outs, dim=1), None, None


# ---------------------------------------------- event processing at step end


def merge_event_grads(events):
    """The orphaned-gradient merge: pre-event spans ran on the OLD cores, so
    their parameters accumulated gradients outside the live set; the shared
    slices are exactly the registrar's copy slices, so each old gradient ADDS
    into the live parameter's shared region (chained events in order -- an
    intermediate core's merged gradients flow onward)."""
    receipts = []
    for ev in events:
        old_key, old_val = ev.heads_old * HEAD_DIM, ev.heads_old * HEAD_V_DIM
        wake = dict(layer=ev.layer_idx, boundary=ev.boundary,
                    heads=ev.heads_old, new_heads=ev.heads_new,
                    fire_smoothed_pr=ev.fire_pr)
        for (name, p_old, p_new, sl) in ev.param_map:
            if name.endswith(O_PROJ):
                g = p_new.grad
                wake["o_proj_new_column_grad_abs_max"] = (
                    float(g[:, old_val:].abs().max())
                    if g is not None and g[:, old_val:].numel() else 0.0)
                gk = p_new.grad
                wake["o_proj_old_column_grad_abs_max"] = (
                    float(gk[:, :old_val].abs().max())
                    if gk is not None and gk[:, :old_val].numel() else 0.0)
            if name.endswith("k_proj.weight"):
                g = p_new.grad
                wake["new_head_k_proj_grad_abs_max"] = (
                    float(g[old_key:].abs().max())
                    if g is not None and g[old_key:].numel() else 0.0)
            if p_old.grad is None:
                continue
            if p_new.grad is None:
                p_new.grad = torch.zeros_like(p_new)
            p_new.grad[sl] += p_old.grad
            p_old.grad = None
        receipts.append(wake)
    return receipts


def _compose(sl1, sl2):
    """Compose two prefix slices (each dim full or a 0-prefix)."""
    out = []
    for s1, s2 in zip(sl1, sl2):
        if s1 == slice(None):
            out.append(s2)
        elif s2 == slice(None):
            out.append(s1)
        else:
            assert s1.stop <= s2.stop, (s1, s2)
            out.append(slice(0, s1.stop))
    return tuple(out)


def rebuild_optimizer(optimizer, model, events, opt_names):
    """AdamW surgery through the event chain: state carried by NAME (stable
    across events) with composed prefix slices; fresh state on the new
    regions -- the smoke's receipted surgery, extended to the chain."""
    if not events:
        return optimizer, opt_names
    carry = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = opt_names.get(id(p))
            if name is not None and name not in carry:
                carry[name] = (p, tuple(slice(None)
                                       for _ in range(p.ndim)))
    for ev in events:
        for (name, p_old, p_new, sl) in ev.param_map:
            prev = carry.get(name)
            carry[name] = ((prev[0], _compose(prev[1], sl)) if prev is not None
                           else (p_old, sl))
    group0 = optimizer.param_groups[0]
    params = [p for p in model.parameters() if p.requires_grad]
    new_opt = torch.optim.AdamW(
        params, lr=group0["lr"], betas=tuple(group0["betas"]),
        weight_decay=group0["weight_decay"], eps=group0["eps"], fused=True)
    new_names = {id(p): n for n, p in model.named_parameters()}
    for p in params:
        name = new_names[id(p)]
        src = carry.get(name)
        if src is None:
            continue
        p_old, composed = src
        st = optimizer.state.get(p_old)
        if not st:
            continue
        new_st = new_opt.state[p]
        new_st["step"] = st.get("step", 0)
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                buf = torch.zeros_like(p)
                buf[composed] = st[key]
                new_st[key] = buf
    return new_opt, new_names


def ema_grow(ema: dict, events) -> list:
    """EMA surgery at growth events: the shared slices' EMA carries; the NEW
    regions' EMA initializes at the birth values (the iterate's honest
    entry point mid-run)."""
    resized = []
    for ev in events:
        for (name, p_old, p_new, sl) in ev.param_map:
            old = ema.get(name)
            if old is not None:
                buf = ev.birth[name].detach().clone()
                buf[sl] = old
                ema[name] = buf
            elif name not in ema:
                ema[name] = ev.birth[name].detach().clone()
            resized.append(name)
    return resized


# ------------------------------------------------------------ the GDN2 builders


def build_gdn2_heads(heads: int, seed: int):
    """The registered lattice_gdn2 arm at a FORCED head count.  heads=2 is
    byte-for-byte ``matched_arms.build_gdn2``'s machine (the same
    construction path); heads=8 is the born-large size control -- built by
    the sanctioned constructor-patch pattern (``byte_velocity.model_for``
    itself patches the constructor), with the capacity check replaced by an
    honest disclosed parameter count (the control is intentionally larger;
    §4: parameter deltas are first-class columns)."""
    from unittest.mock import patch
    from experiments.expressivity_tasks.x42_pile_scale import (
        velocity_baselines as BV)
    from experiments.expressivity_tasks.x40_emender_lm import (
        part8_surface, scale_surface)
    from ndm.models import external_gdn2
    from fla.modules import FusedLinearCrossEntropyLoss

    matched_arms.configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, policy, nom = L.build_policy("lattice_gdn2")
    original = external_gdn2.GDN2ExternalMLPLayer

    def forced(*args, **kwargs):
        kwargs.pop("num_heads", None)
        kwargs["expansion"] = policy.expansion
        kwargs["num_heads"] = heads
        return original(*args, **kwargs)

    class _AtHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with patch.object(BV, "VOCAB", 256), \
                    patch.object(scale_surface, "VOCAB", 256), \
                    patch.object(part8_surface, "VOCAB", 256), \
                    patch.object(external_gdn2, "GDN2ExternalMLPLayer", forced):
                part8_surface.gdn_config.cache_clear()
                self.model = part8_surface.GDNModel("150M", BV.CAPACITY)
            with torch.no_grad():
                for module in self.model.modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                        module.weight.mul_(policy.init_scale)
                for block in self.model.blocks:
                    core = block.layer.gdn2.gdn2
                    core.A_log.add_(math.log(policy.decay_rate_scale))
                    core.dt_bias.add_(policy.dt_bias_shift)
            self.criterion = FusedLinearCrossEntropyLoss()
            self.arm = "gdn2"

        def loss(self, ids, targets):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden, _ = BV.features(self.model, ids, self.arm, True)
                return self.criterion(hidden.to(torch.bfloat16).contiguous(),
                                      targets.contiguous(),
                                      self.model.head.weight.to(torch.bfloat16))

    model = _AtHeads()
    optimizer_config = dict(lr=policy.lr, betas=(0.9, policy.beta2),
                           weight_decay=policy.weight_decay,
                           warmup_steps=policy.warmup_steps,
                           microbatch=policy.microbatch)
    total = sum(p.numel() for p in model.parameters())
    summary = dict(base_arm="lattice_gdn2", forced_heads=heads,
                   parameters=dict(total=total,
                                   note=("the registered arm verbatim"
                                         if heads == 2 else
                                         "the born-large size control: growth "
                                         "adds parameters, so the control is "
                                         "intentionally larger -- the delta "
                                         "is a first-class column (§4)")),
                   nomination={k: nom[k] for k in
                               ("arm", "receipt", "receipt_sha256",
                                "generation", "candidate", "search_score")},
                   policy=dict(lr=policy.lr, microbatch=policy.microbatch,
                               beta2=policy.beta2,
                               weight_decay=policy.weight_decay,
                               warmup_steps=policy.warmup_steps,
                               init_scale=policy.init_scale,
                               decay_rate_scale=policy.decay_rate_scale,
                               dt_bias_shift=policy.dt_bias_shift,
                               expansion=policy.expansion))
    return model, optimizer_config, summary


def install_growth(model, sink):
    """Wrap every block's GatedDeltaNet2 core with the growth wrapper (the
    registrar pattern; GDN2's code read-only).  Parameter NAMES stay stable
    across events (the wrapper keeps its ``core`` slot), which is what makes
    the optimizer/EMA carry by name legal.

    The growth arm also runs its features pass WITHOUT activation
    checkpointing: a growth event mutates the live module MID-FORWARD, and
    reentrant-checkpoint recomputation would re-run the block forward a
    second time against a different module (and against the probe's
    inference-mode tensors) -- an inconsistent backward.  Checkpointing is
    a bit-identical MEMORY KNOB on the fixed path (the fixed arms keep the
    banked checkpointed path verbatim); the growth arm pays the honest
    activation memory instead."""
    import types
    from experiments.expressivity_tasks.x42_pile_scale import (
        velocity_baselines as BV)
    wrappers = []
    for idx, block in enumerate(model.model.blocks):
        ext = block.layer.gdn2
        core = ext.gdn2
        wrapper = GrowingGDN2Core(core, idx, sink)
        ext.gdn2 = wrapper
        wrappers.append(wrapper)

    def loss_no_checkpoint(self, ids, targets):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, _ = BV.features(self.model, ids, self.arm, False)
            return self.criterion(hidden.to(torch.bfloat16).contiguous(),
                                  targets.contiguous(),
                                  self.model.head.weight.to(torch.bfloat16))

    model.loss = types.MethodType(loss_no_checkpoint, model)
    # resolve every wrapper's core-name prefix from the live module tree so
    # events carry MODEL-LEVEL parameter names (the EMA/optimizer state maps
    # are keyed by model-level names; core-relative names would collide
    # across the 27 mixers)
    by_id = {id(p): n for n, p in model.named_parameters()}
    for wrapper in wrappers:
        rel = next(iter(dict(wrapper.core.named_parameters())))
        first = next(iter(dict(wrapper.core.named_parameters()).values()))
        wrapper.name_prefix = by_id[id(first)][:-len(rel)]
        assert wrapper.name_prefix.endswith(".core.") or \
            wrapper.name_prefix.endswith(".gdn2."), wrapper.name_prefix
    return wrappers


# ------------------------------------------------------- saturation diagnostics


def gdn2_saturation_diagnostics(model, windows, ctx, pool, tau=TAU_SAT):
    """Per-layer per-boundary PR (the registered saturation statistic, exact
    spectra via the sidecar) and the ι co-diagnostic, on the model's CURRENT
    state, from the real kernel's own trajectory (forward hooks; the banked
    k8 probe_diagnostics pattern)."""
    store = []
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, _output):
            x = args[0]
            core = module.core if isinstance(module, GrowingGDN2Core) \
                else module
            with torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                q, k, v, g, b, w = _kernel_inputs(core, x)
                _o, _s, h = _chunk_call(core, q, k, v, g, b, w,
                                        output_final_state=True,
                                        return_intermediate_states=True)
                pr = pr_from_trajectory(pool, h).cpu()      # [B,C-1,H]
                khat = F.normalize(
                    k.reshape(-1, core.num_heads, HEAD_DIM).float(),
                    dim=-1).reshape(k.shape)
                iota = iota_from_trajectory(h, b, khat).cpu()  # [B,C,H]
            store.append(dict(layer=layer_idx, heads=core.num_heads,
                              pr=pr.numpy(), iota=iota.numpy()))
        return hook

    for idx, block in enumerate(model.model.blocks):
        handles.append(block.layer.gdn2.gdn2.register_forward_hook(
            make_hook(idx)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in windows:
                x, _y, _v = next(grouped([row], 1, context=ctx))
                model.loss(x, _y)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    per_layer, curves_pr, curves_iota = {}, [], []
    n_layers = len(model.model.blocks)
    for idx in range(n_layers):
        rows = [r for r in store if r["layer"] == idx]
        pr = np.concatenate([r["pr"].reshape(-1, r["pr"].shape[-1])
                             for r in rows])             # [n, H]
        iota = np.concatenate([r["iota"].reshape(-1, r["iota"].shape[-1])
                               for r in rows])
        boundary_pr = np.stack([r["pr"].mean(axis=(0, 2)) for r in rows])
        boundary_iota = np.stack([r["iota"].mean(axis=(0, 2)) for r in rows])
        smoothed_max = float(max(smoothed_series(s).max()
                                 for s in boundary_pr))
        per_layer[idx] = dict(
            heads=rows[0]["heads"],
            pr_mean=float(pr.mean()), pr_min=float(pr.min()),
            pr_max=float(pr.max()),
            pr_smoothed_max=smoothed_max,
            fires=int(sum(int((smoothed_series(s) >= tau).sum())
                          for s in boundary_pr)),
            iota_mean=float(iota.mean()), iota_max=float(iota.max()))
        curves_pr.append(boundary_pr.mean(axis=0).tolist())
        curves_iota.append(boundary_iota.mean(axis=0).tolist())
    agg_pr = np.concatenate([r["pr"].reshape(-1, r["pr"].shape[-1])
                             for r in store])
    return dict(
        windows=len(windows), context=ctx, tau=tau,
        regime="eval, bf16 autocast, the kernel's own per-chunk trajectory",
        pr=dict(mean=float(agg_pr.mean()), min=float(agg_pr.min()),
                max=float(agg_pr.max()),
                per_head_heterogeneity="receipted per layer below"),
        per_layer=per_layer,
        boundary_curve_pr_mean_over_layers=np.mean(curves_pr, axis=0).tolist(),
        boundary_curve_iota_mean_over_layers=np.mean(curves_iota,
                                                     axis=0).tolist(),
    )



# ------------------------------------------------ emender-arm native diagnostics


def emender_diagnostics(model, windows, ctx):
    """The K8 lane's own saturation/interference statistics at the regime's
    context (occupancy, read entropy, op rates, growth events -- the banked
    ``k8_probe_growth.probe_diagnostics`` pattern, imported machinery)."""
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped as _grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        OP_NAMES, op_histogram)
    from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
        growing_scan)
    from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
    per_layer = []
    handles = []

    def make_hook(layer_index, mixer):
        def hook(_module, args, _output):
            x = args[0]
            batch, tokens, _ = x.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                query = mixer.query(x).view(batch, tokens, mixer.heads,
                                            mixer.value_dim)
                payload = mixer.value(x).view(batch, tokens, mixer.heads,
                                             mixer.value_dim)
                decay = torch.sigmoid(mixer.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                result = growing_scan(
                    query, payload, decay, policy=mixer.policy,
                    start_rank=mixer.rank, rank_cap=mixer.rank_cap,
                    segment_tokens=mixer.segment_tokens, collect=True)
            with torch.no_grad():
                lane0, hist = Counter(), Counter()
                decisions = occ_n = 0
                occ_sum = ent_sum = ent_n = 0
                for span in result.collected:
                    schedule = span["schedule"]
                    lane0.update({OP_NAMES[i]: int(torch.bincount(
                        schedule.op[..., 0].reshape(-1).to("cpu"),
                        minlength=len(OP_NAMES))[i])
                        for i in range(len(OP_NAMES))})
                    hist.update(op_histogram(schedule))
                    decisions += int(batch * (span["t1"] - span["t0"])
                                     * mixer.heads)
                    occ = span["live_pre"].float().sum(-1)
                    occ_sum += float(occ.sum()); occ_n += occ.numel()
                    w = span["weights"]; mass = w.sum(-1); has = mass > 0
                    probs = torch.where(has.unsqueeze(-1),
                                        w / mass.clamp_min(1e-30).unsqueeze(-1),
                                        torch.zeros_like(w))
                    contrib = torch.where(probs > 0, probs * probs.log(),
                                          torch.zeros_like(probs))
                    ent_sum += float(contrib.sum(-1)[has].sum())
                    ent_n += int(has.sum())
                per_layer.append((layer_index, dict(
                    lane0=dict(lane0), hist=dict(hist), decisions=decisions,
                    occ_sum=occ_sum, occ_n=occ_n,
                    ent_sum=ent_sum, ent_n=ent_n,
                    final_rank=result.rank,
                    events=[dict(e) for e in result.stats.events])))
        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.mixer.register_forward_hook(
            make_hook(index, block.mixer)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in windows:
                x, _y, _v = next(_grouped([row], 1, context=ctx))
                model.loss(x, _y)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    layers = sorted({i for i, _ in per_layer})
    by_layer, occ_sum, occ_n, ent_sum, ent_n = {}, 0, 0, 0.0, 0
    decisions = sum(s["decisions"] for _, s in per_layer)
    hist = Counter()
    events_total = 0
    span_tables = Counter()
    for layer in layers:
        stats = [s for i, s in per_layer if i == layer]
        by_layer[layer] = dict(
            op_rate_per_token={OP_NAMES[i]:
                               sum(s["lane0"][OP_NAMES[i]] for s in stats)
                               / max(1, sum(s["decisions"] for s in stats))
                               for i in range(len(OP_NAMES))},
            final_rank=stats[0]["final_rank"],
            growth_events=stats[0]["events"],
            occupancy_mean_live_slots=float(
                sum(s["occ_sum"] for s in stats)
                / max(1, sum(s["occ_n"] for s in stats))),
            read_weight_entropy_nats=float(
                sum(s["ent_sum"] for s in stats)
                / max(1, sum(s["ent_n"] for s in stats))))
        occ_sum += sum(s["occ_sum"] for s in stats)
        occ_n += sum(s["occ_n"] for s in stats)
        ent_sum += sum(s["ent_sum"] for s in stats)
        ent_n += sum(s["ent_n"] for s in stats)
        for s in stats:
            hist.update(s["hist"])
            events_total += len(s["events"])
            span_tables[tuple((sp["t0"], sp["t1"], sp["rank"])
                              for sp in s.get("spans", []))] += 1
    return dict(
        windows=len(windows), context=ctx,
        regime="eval, bf16 autocast, the growing collect walk the arm trains on",
        op_rate_per_token_lane0={OP_NAMES[i]:
                                 sum(s["lane0"][OP_NAMES[i]] for _, s in per_layer)
                                 / max(1, decisions)
                                 for i in range(len(OP_NAMES))},
        op_histogram_all_lanes=dict(hist),
        occupancy=dict(mean_live_slots=float(occ_sum / max(1, occ_n)),
                       rank_cap=2048),
        read_weight_entropy=dict(mean=float(ent_sum / max(1, ent_n))),
        growth=dict(events_per_probe=events_total / max(1, len(layers)),
                    per_layer={k: v["growth_events"] for k, v in
                              by_layer.items()}),
        per_layer=by_layer)


def growth_snapshot(model) -> dict:
    """The live head-count state of every wrapped GDN2 mixer (cheap,
    host-side, per step)."""
    final = Counter()
    events = 0
    for wrapper in model.model.blocks:
        w = wrapper.layer.gdn2.gdn2
        final[w.num_heads] += 1
        events += w.n_events
    return dict(mixers=len(final), heads_histogram={str(k): v for k, v in
                                                    sorted(final.items())},
                total_events_ever=events)


# ------------------------------------------------------------ the training loop


def train_c3(label, model, opt_cfg, stream, ctx, machine, out, pool,
             grow_sink=None, smoke=False, micro_override=None):
    """The corrected-protocol loop (the banked probe protocol + the 24-step
    cooldown): matched 20M exposure, honest clock, the protocol's own grad
    clip, EMA over the constant phase, CPU snapshots at the checkpoint grid,
    growth-event processing (grad merge -> optimizer surgery -> EMA surgery)
    between backward and the optimizer step.  Returns (run, log_rows)."""
    batch = 2 if smoke else windows_per_step(ctx)
    micro = micro_override if micro_override else microbatch_for(ctx, machine)
    if smoke:
        micro = min(micro, 1)
    warmup = opt_cfg["warmup_steps"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"],
                                  betas=tuple(opt_cfg["betas"]),
                                  weight_decay=opt_cfg["weight_decay"],
                                  fused=True)
    opt_names = {id(p): n for n, p in model.named_parameters()}
    ema = {n: p.detach().float().clone()
           for n, p in model.named_parameters() if p.requires_grad}
    batches = train_batches(stream, ctx, batch)
    elapsed = 0.0
    step = seen = 0
    rows = []
    snapshots = {}
    wake_receipts = []
    pending_wake = []      # events of the previous step (second-step receipt)
    constant_steps = None
    targets = int(3 * batch * ctx) if smoke else EXPOSURE_TARGETS
    cooldown_steps = 2 if smoke else COOLDOWN_STEPS
    diverged = False
    log = (out / "training.jsonl").open("w", buffering=1)

    def do_step(cooldown_k):
        nonlocal elapsed, step, seen, diverged, optimizer, opt_names
        nonlocal pending_wake
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            rows_batch = next(batches)
        except StopIteration:
            raise RuntimeError("train split exhausted before the budget")
        n = sum(len(r[3]) for r in rows_batch)
        optimizer.zero_grad(set_to_none=True)
        factor = min(1., (step + 1) / max(1, warmup))
        if cooldown_k:
            factor *= cooldown_factor(cooldown_k, cooldown_steps)
        lr = opt_cfg["lr"] * factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        if grow_sink is not None:
            grow_sink["events"] = []
        loss_sum = 0.0
        for x, y, valid in grouped(rows_batch, micro, context=ctx):
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if diverged:
            elapsed += time.monotonic() - start
            return None
        growth_row = None
        probe_row = None
        if grow_sink is not None and pending_wake:
            # the second-step wake-up receipt: the previous step's events'
            # new-head internals must now receive nonzero gradient (the
            # o_proj columns moved at the first post-event step)
            for receipt, ev in pending_wake:
                for (name, _p_old, p_new, sl) in ev.param_map:
                    if name.endswith("k_proj.weight") and p_new.grad \
                            is not None:
                        receipt["second_step_new_head_k_proj_grad_abs_max"] = \
                            float(p_new.grad[ev.heads_old * HEAD_DIM:]
                                  .abs().max())
                        break
            pending_wake = []
        if grow_sink is not None and grow_sink["events"]:
            events = grow_sink["events"]
            receipts = merge_event_grads(events)
            wake_receipts.extend(receipts)
            pending_wake = list(zip(receipts, events))
            optimizer, opt_names = rebuild_optimizer(optimizer, model,
                                                    events, opt_names)
            ema_grow(ema, events)
            growth_row = dict(events=len(events),
                              heads=growth_snapshot(model)["heads_histogram"],
                              boundaries=[ev.boundary for ev in events],
                              layers=[ev.layer_idx for ev in events])
        if grow_sink is not None:
            spans = [s for w in (b.layer.gdn2.gdn2
                                 for b in model.model.blocks)
                     if getattr(w, "last_probe", None)
                     and w.core.num_heads < w.cap
                     for s in w.last_probe["spans"]]
            if spans:
                probe_row = dict(
                    pr_smoothed_max=max(s["pr_smoothed_max"]
                                        for s in spans),
                    pr_mean=sum(s["pr_mean"] for s in spans) / len(spans),
                    iota_mean=sum(s["iota_mean"] for s in spans)
                    / len(spans))
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                       error_if_nonfinite=True)
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed += time.monotonic() - start
        step += 1
        seen += n
        row = dict(step=step, active_seconds=elapsed, bytes=seen,
                   loss=loss_sum, lr_factor=factor,
                   phase="cooldown" if cooldown_k else "constant",
                   time_ns=time.time_ns(), growth=growth_row,
                   probe=probe_row)
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        if not cooldown_k:                      # the EMA: constant phase only
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        if name not in ema:
                            ema[name] = p.detach().float().clone()
                        else:
                            ema[name].mul_(EMA_DECAY).add_(p.detach().float(),
                                                           alpha=1 - EMA_DECAY)
        if step in CHECKPOINT_STEPS or (smoke and step <= 3):
            snapshots[step] = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        if step % 10 == 0 or step <= 2:
            extra = ""
            if grow_sink is not None:
                snap = growth_snapshot(model)
                extra = (f" heads {snap['heads_histogram']} "
                         f"events {snap['total_events_ever']}")
            print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                  f"loss {loss_sum:.4f}{extra}", flush=True)
        return row

    try:
        while seen < targets:
            if do_step(0) is None:
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            constant_steps=step, ema=ema,
                            wake_receipts=wake_receipts)
        constant_steps = step
        snapshots[constant_steps] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        for k in range(1, cooldown_steps + 1):
            if do_step(k) is None:
                break
        snapshots["final"] = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    finally:
        log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                constant_steps=constant_steps, rows=rows,
                snapshots=snapshots, ema=ema, wake_receipts=wake_receipts)


# ----------------------------------------------------------------- scoring


def score_arm(model, run, eval_windows, ctx, label):
    """The three scorings: the constant-LR curve (banked comparability),
    the post-cooldown eval (PRIMARY), the EMA iterate at the constant-phase
    end (SECONDARY, ranking-only).  Restored snapshots; training is over."""
    rows_by_step = {r["step"]: r for r in run["rows"]}
    constant_curve = []
    skipped = []
    for step in sorted(s for s in run["snapshots"] if s != "final"):
        try:
            model.load_state_dict(run["snapshots"][step])
        except RuntimeError as exc:
            skipped.append(dict(step=step, reason=(
                "the parameter set grew past this checkpoint (a growth event "
                "changed shapes); the train-loss row stands" if
                "growth" in label else repr(exc))))
            continue
        result = evaluate_ctx(model, eval_windows, ctx)
        constant_curve.append(dict(kind="constant", step=step,
                                   bytes=rows_by_step[step]["bytes"],
                                   train_loss=rows_by_step[step]["loss"],
                                   eval_nll=result["score"],
                                   eval_windows=result["eval_windows"],
                                   eval_targets=result["eval_targets"]))
        print(f"  {label} constant step {step}: eval "
              f"{result['score']:.4f}", flush=True)
    primary = None
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        result = evaluate_ctx(model, eval_windows, ctx)
        primary = dict(kind="post-cooldown (PRIMARY)", step=run["step"],
                       eval_nll=result["score"],
                       eval_windows=result["eval_windows"],
                       eval_targets=result["eval_targets"])
        print(f"  {label} POST-COOLDOWN (primary): {result['score']:.4f}",
              flush=True)
    secondary = None
    if run["ema"] is not None and run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in run["ema"]:
                    p.copy_(run["ema"][name].to(p.dtype))
        result = evaluate_ctx(model, eval_windows, ctx)
        secondary = dict(kind="EMA iterate at the constant-phase end "
                              "(SECONDARY, ranking-only)",
                         step=run["constant_steps"],
                         eval_nll=result["score"],
                         eval_windows=result["eval_windows"],
                         eval_targets=result["eval_targets"])
        print(f"  {label} EMA (secondary): {result['score']:.4f}", flush=True)
    return dict(constant_curve=constant_curve, primary=primary,
                secondary=secondary, skipped_evals=skipped)


def diagnostics_for_snapshots(model, run, diag_windows, ctx, pool, machine,
                             label, steps=("constant-final", "post-cooldown")):
    """The saturation/interference divergence receipts on the trained
    states (shape-compatible snapshots only)."""
    out = {}
    for tag in steps:
        key = run["constant_steps"] if tag == "constant-final" else "final"
        if key not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][key])
        if machine == "emender":
            out[tag] = emender_diagnostics(model, diag_windows, ctx)
        else:
            out[tag] = gdn2_saturation_diagnostics(model, diag_windows, ctx,
                                                    pool)
        print(f"  {label} diagnostics[{tag}]: PR mean "
              f"{out[tag]['pr']['mean']:.1f}" if machine != "emender" else
              f"  {label} diagnostics[{tag}]: occupancy "
              f"{out[tag]['occupancy']['mean_live_slots']:.1f}", flush=True)
    return out


# --------------------------------------------------------------- subcommands


def cmd_kshapes(args) -> dict:
    """Fresh kernel/shape receipts at the regime's long horizons (the ruling's
    item 4: receipt, escalate on impossibility, never patch)."""
    from ndm.models import external_gdn2
    from ndm.models.external_gdn2 import GDN2ExternalMLPLayer
    external_gdn2._load_gdn2_class()
    chunk_gdn2 = external_gdn2._GDN2_OPS_MODULE.chunk_gdn2
    require_lease()
    torch.manual_seed(SEED)
    receipt = dict(probe="gdn2_growth_c3 kshapes", seed=SEED,
                   env=dict(NDM_PIN_TRITON_AUTOTUNE=os.environ.get(
                       "NDM_PIN_TRITON_AUTOTUNE")))
    ok = True
    for heads in (2, 4, 8):
        torch.manual_seed(SEED + heads)
        layer = GDN2ExternalMLPLayer(dim=576, expansion=2., head_dim=128,
                                     num_heads=heads, use_conv=True, d_conv=4,
                                     gdn2_mlp_ratio=2.0).cuda().train()
        core = layer.gdn2.gdn2
        for T, B in ((8192, 2), (16384, 1)):
            x = torch.randn(B, T, 576, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(x)[0]
                out.pow(2).sum().backward()
            torch.cuda.synchronize()
            wall = time.time() - t0
            finite = all(torch.isfinite(p.grad).all().item()
                         for p in layer.parameters() if p.grad is not None)
            with torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                q, k, v, g, b, w = _kernel_inputs(core, x)
                _o, _s, h = _chunk_call(core, q, k, v, g, b, w,
                                        output_final_state=True,
                                        return_intermediate_states=True)
            row = dict(T=T, B=B, heads=heads,
                       fwd_bwd_wall_s=round(wall, 2), grads_finite=finite,
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       trajectory_shape=list(h.shape),
                       trajectory_finite=bool(torch.isfinite(h).all()))
            receipt[f"H{heads}_T{T}"] = row
            ok = ok and finite and row["trajectory_finite"]
            del x, out
            torch.cuda.empty_cache()
        del layer
        torch.cuda.empty_cache()
    receipt["verdict"] = ("no kernel/shape impossibility at the regime's "
                          "horizons on the heads axis" if ok else
                          "IMPOSSIBILITY -- ESCALATE")
    return receipt


def cmd_calibrate(args) -> dict:
    """The τ_sat calibration scan (the registration §5's disclosed protocol):
    the FIXED GDN2 arm's per-boundary PR receipted on the C3 stream at BOTH
    candidate horizons (the trained-machine anchors), the ι co-diagnostic
    alongside per boundary; then the regime lock (the SHORTEST horizon at
    which the registered 96 is reachable)."""
    card = require_lease()
    pool = SPECTRA.SpectraPool()
    scans = {}
    try:
        for ctx in HORIZONS:
            identity, stream = c3_stream(ctx)
            elig = eligible_documents(stream, ctx)
            eval_windows = full_windows(stream, "calibration", ctx, EVAL_WINDOWS)
            diag_windows = eval_windows[:DIAG_WINDOWS]
            out = RUNS / f"calibrate_{ctx}_seed{SEED}"
            out.mkdir(parents=True, exist_ok=False)
            model, opt_cfg, summary = matched_arms.build_gdn2(SEED)
            model = model.cuda().train()
            config = dict(probe="gdn2_growth_c3 calibrate", context=ctx,
                          arm="gdn2_fixed (H=2, the registered arm verbatim)",
                          seed=SEED, exposure_targets=EXPOSURE_TARGETS,
                          windows_per_step=windows_per_step(ctx),
                          microbatch=microbatch_for(ctx, "gdn2"),
                          cooldown_steps=COOLDOWN_STEPS,
                          optimizer=opt_cfg, summary=summary,
                          identity=identity, eligible=elig,
                          protocol="the standing probe rule transplanted: "
                                   "20M matched-exposure landing, "
                                   f"{windows_per_step(ctx)} full-{ctx} "
                                   "windows/step, 24-step linear cooldown "
                                   "(13.7% of the constant phase)",
                          bindings=regime_bindings(),
                          git=matched_arms.git_state())
            atomic_json(out / "start.json", dict(
                config, config_sha256=matched_arms.config_sha256(config),
                pid=os.getpid(), session=matched_arms.session_meta(card),
                started_ns=time.time_ns(),
                parameters=dict(total=sum(p.numel()
                                         for p in model.parameters()))))
            run = train_c3(f"calibrate_{ctx}", model, opt_cfg, stream, ctx,
                           "gdn2", out, pool)
            if run["status"] != "completed":
                raise RuntimeError(f"calibration at {ctx} {run['status']}")
            scores = score_arm(model, run, eval_windows, ctx, f"calib_{ctx}")
            model.load_state_dict(run["snapshots"]["final"])
            diag_final = gdn2_saturation_diagnostics(model, diag_windows,
                                                      ctx, pool)
            diag_const = None
            if run["constant_steps"] in run["snapshots"]:
                model.load_state_dict(run["snapshots"][run["constant_steps"]])
                diag_const = gdn2_saturation_diagnostics(model, diag_windows,
                                                         ctx, pool)
            fires_const = sum(v["fires"] for v in diag_const["per_layer"].values()) \
                if diag_const else 0
            fires_final = sum(v["fires"] for v in diag_final["per_layer"].values())
            receipt = dict(
                probe="gdn2_growth_c3 calibration", context=ctx,
                arm="gdn2_fixed (the registered 150M arm, H=2)",
                seed=SEED,
                tau_sat_registered=TAU_SAT,
                reachability_rule="tau_sat=96 is REACHABLE at a horizon iff "
                                  "the trained (constant-phase-final) fixed "
                                  "arm's smoothed mean PR fires at >=96 on "
                                  "some (layer, window, boundary)",
                tau_reachable=bool(fires_const > 0),
                fires=dict(constant_phase_final=fires_const,
                           post_cooldown=fires_final),
                pr=dict(
                    constant_phase_final=dict(
                        mean=diag_const["pr"]["mean"],
                        min=diag_const["pr"]["min"],
                        max=diag_const["pr"]["max"],
                        per_layer_smoothed_max={
                            str(k): v["pr_smoothed_max"]
                            for k, v in diag_const["per_layer"].items()}
                    ) if diag_const else None,
                    post_cooldown=dict(
                        mean=diag_final["pr"]["mean"],
                        min=diag_final["pr"]["min"],
                        max=diag_final["pr"]["max"],
                        per_layer_smoothed_max={
                            str(k): v["pr_smoothed_max"]
                            for k, v in diag_final["per_layer"].items()}),
                    boundary_curve_mean_over_layers_post_cooldown=
                        diag_final["boundary_curve_pr_mean_over_layers"]),
                iota=dict(
                    post_cooldown=dict(
                        mean=float(np.mean([v["iota_mean"] for v in
                                            diag_final["per_layer"].values()])),
                        max=max(v["iota_max"] for v in
                                diag_final["per_layer"].values())),
                    boundary_curve_mean_over_layers_post_cooldown=
                        diag_final["boundary_curve_iota_mean_over_layers"]),
                eligible_documents=elig,
                eval_windows=len(eval_windows),
                scores=scores,
                exposure_bytes=run["seen"], exposure_steps=run["step"],
                constant_steps=run["constant_steps"],
                walls=dict(train_seconds=round(run["elapsed"], 1)),
                spectra_pool=pool.stats(),
                config_sha256=matched_arms.config_sha256(config),
                config=config,
                trajectory=matched_arms.trajectory_sample(
                    run["rows"], keep_first=5, keep_last=5, stride=25),
            )
            atomic_json(RECEIPTS / f"gdn2_growth_c3_calibration_{ctx}.json",
                        receipt)
            scans[ctx] = receipt
            print(json.dumps(dict(context=ctx,
                                  tau_reachable=receipt["tau_reachable"],
                                  fires=fires_const,
                                  pr_mean=diag_final["pr"]["mean"]),
                             indent=1), flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        pool_stats = pool.stats()
        pool.close()

    reachable = [ctx for ctx in HORIZONS if scans[ctx]["tau_reachable"]]
    if not reachable:
        lock = dict(
            locked=False,
            escalation="NEITHER horizon makes tau_sat=96 reachable on the "
                       "trained fixed machine -- per the ruling this receipt "
                       "IS the escalation artifact; the arms do not run "
                       "until the supervisor rules (the calibrated-"
                       "threshold branch of §5, or the informative null)",
            scans={str(ctx): dict(
                tau_reachable=scans[ctx]["tau_reachable"],
                pr=dict(mean=scans[ctx]["pr"]["post_cooldown"]["mean"],
                        max=scans[ctx]["pr"]["post_cooldown"]["max"],
                        per_layer_smoothed_max=scans[ctx]["pr"]
                        ["post_cooldown"]["per_layer_smoothed_max"]),
                receipt=f"receipts/gdn2_growth_c3_calibration_{ctx}.json")
                for ctx in HORIZONS},
            predictions=PREDICTIONS, spectra_pool=pool_stats,
            bindings=regime_bindings(), git=matched_arms.git_state())
        atomic_json(RECEIPTS / "gdn2_growth_c3_regime.json", lock)
        print("ESCALATION: tau_sat=96 unreachable at both horizons; the "
              "regime receipt is the escalation artifact", flush=True)
        return lock
    ctx = min(reachable)
    lock = dict(
        locked=True, context=ctx,
        tau_sat=TAU_SAT,
        tau_decision="the registered 96 stands (reachable at the locked "
                     "horizon on the trained fixed machine)",
        horizon_rule="the SHORTEST horizon at which the registered tau_sat "
                     "is reachable (the supervisor's ruling)",
        scanned={str(c): dict(tau_reachable=scans[c]["tau_reachable"],
                              fires=scans[c]["fires"],
                              receipt=f"receipts/"
                                      f"gdn2_growth_c3_calibration_{c}.json")
                 for c in HORIZONS},
        protocol=dict(exposure_targets=EXPOSURE_TARGETS,
                      windows_per_step=windows_per_step(ctx),
                      landing_targets=175 * windows_per_step(ctx) * ctx,
                      constant_steps_expected=175,
                      cooldown_steps=COOLDOWN_STEPS,
                      cooldown_fraction=round(COOLDOWN_STEPS / 175, 4),
                      segment=SEGMENT, head_cap=HEAD_CAP, smooth=SMOOTH,
                      seed=SEED,
                      eval_windows=EVAL_WINDOWS,
                      microbatch=dict(gdn2=microbatch_for(ctx, "gdn2"),
                                      emender=microbatch_for(ctx, "emender")),
                      stream="the registered flagship stream's documents, "
                             "full %d-byte windows from eligible documents "
                             "only (byte_length >= %d), document-bounded, "
                             "split-disjoint" % (ctx, ctx + 1),
                      optimizer="plain AdamW at the registered gdn2 "
                                "nomination's transferred settings, all four "
                                "arms (the K6/K8/side-by-side convention; "
                                "identical numbers on both machines)"),
        predictions=PREDICTIONS, spectra_pool=pool_stats,
        eligible_documents=scans[ctx]["eligible_documents"],
        bindings=regime_bindings(), git=matched_arms.git_state())
    atomic_json(RECEIPTS / "gdn2_growth_c3_regime.json", lock)
    print(json.dumps(dict(locked=True, context=ctx, tau_sat=TAU_SAT),
                     indent=1), flush=True)
    return lock


def regime_bindings() -> dict:
    return dict(
        registration=str(GROWTH_REG.relative_to(ROOT.parents[2])),
        registration_sha256=digest(GROWTH_REG),
        growth_smoke_receipt=str(GROWTH_SMOKE_RECEIPT.relative_to(ROOT.parents[2])),
        growth_smoke_sha256=digest(GROWTH_SMOKE_RECEIPT),
        spectra_sidecar_selftest=str(SIDECAR_SELFTEST.relative_to(ROOT.parents[2])),
        spectra_sidecar_selftest_sha256=digest(SIDECAR_SELFTEST),
        lattice_read_only=True, gdn2_code_read_only=True)


def cmd_arm(args) -> dict:
    """One C3 arm at the locked regime.  The growth arms refuse to run
    without the committed regime lock (the calibration gates everything
    downstream)."""
    card = require_lease()
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}; expected {ARMS}")
    if args.smoke:
        # the machinery shakedown runs BEFORE the regime lock (it is not an
        # arm; tiny exposure, forced tau, never scored)
        ctx, tau = 8192, TAU_SAT
    else:
        lock = json.loads((RECEIPTS / "gdn2_growth_c3_regime.json").read_text())
        if not lock.get("locked"):
            raise RuntimeError("the regime is not locked (the calibration's "
                               "escalation branch): the arms do not run")
        ctx, tau = lock["context"], lock["tau_sat"]
    identity, stream = c3_stream(ctx)
    eval_windows = full_windows(stream, "calibration", ctx, EVAL_WINDOWS)
    diag_windows = eval_windows[:DIAG_WINDOWS]
    pool = SPECTRA.SpectraPool()
    try:
        sink = {"pool": pool, "events": []}
        machine = "emender" if args.arm == "emender_growable" else "gdn2"
        if args.arm == "gdn2_fixed":
            model, opt_cfg, summary = matched_arms.build_gdn2(args.seed)
        elif args.arm == "gdn2_fixed_8":
            model, opt_cfg, summary = build_gdn2_heads(8, args.seed)
        elif args.arm == "gdn2_growth":
            model, opt_cfg, summary = matched_arms.build_gdn2(args.seed)
        else:
            from experiments.expressivity_tasks.x45_matrix_emender.k8_growth \
                import convert_model_to_growing
            model, opt_cfg, summary = matched_arms.build_matrix(
                "matrix_ops", 128, args.seed, k5b=False)
            model = convert_model_to_growing(
                model, segment_tokens=SEGMENT, rank_cap=2048, grow=True,
                start_rank=128)
            summary = dict(summary)
            summary["scan_path"] = ("k8_growth verbatim (start rank 128, the "
                                    "registered trigger, cap 2048)")
        model = model.cuda().train()
        grow_sink = None
        if args.arm == "gdn2_growth":
            install_growth(model, sink)
            grow_sink = sink
        label = args.arm if not args.smoke else f"{args.arm}_smoke"
        out = RUNS / f"{label}_seed{args.seed}"
        out.mkdir(parents=True, exist_ok=False)
        config = dict(
            probe="gdn2_growth_c3", arm=args.arm, seed=args.seed,
            context=ctx, tau_sat=tau, exposure_targets=EXPOSURE_TARGETS,
            windows_per_step=windows_per_step(ctx),
            microbatch=(args.microbatch if args.microbatch
                        else microbatch_for(ctx, machine)),
            cooldown_steps=COOLDOWN_STEPS, optimizer=opt_cfg, summary=summary,
            identity=identity,
            smoke=bool(args.smoke),
            smoke_tau=(args.smoke_tau if args.smoke else None),
            regime_lock=(str((RECEIPTS / "gdn2_growth_c3_regime.json")
                             .relative_to(ROOT.parents[2]))
                         if not args.smoke else
                         "NOT LOCKED YET (the shakedown precedes the "
                         "calibration; never scored)"),
            state_columns=dict(
                per_layer_state=(f"heads x 128x256 = {2 * HEAD_V_DIM}"
                                 if machine == "gdn2" else
                                 "slots x value_dim, rank cap 2048")),
            bindings=dict(regime_bindings(),
                          regime_lock_sha256=(digest(
                              RECEIPTS / "gdn2_growth_c3_regime.json")
                              if not args.smoke else None)),
            git=matched_arms.git_state(),
            stopping_rule=("matched-EXPOSURE: 20M targets (full-%d windows), "
                           "constant LR, then the 24-step linear cooldown"
                           % ctx))
        atomic_json(out / "start.json", dict(
            config, config_sha256=matched_arms.config_sha256(config),
            pid=os.getpid(), session=matched_arms.session_meta(card),
            started_ns=time.time_ns(),
            parameters=dict(total=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for p in
                                          model.parameters()
                                          if p.requires_grad))))
        print(json.dumps(dict(arm=args.arm, context=ctx, card=card,
                              seed=args.seed,
                              parameters=sum(p.numel()
                                             for p in model.parameters()))),
              flush=True)
        if args.smoke and args.arm == "gdn2_growth":
            # the shakedown forces a reachable tau so the event machinery
            # (probe, registrar, merge, surgery, EMA resize) executes; a
            # SMOKE-ONLY knob, receipted, never used by the arms
            for wrapper in [b.layer.gdn2.gdn2 for b in model.model.blocks]:
                wrapper.tau = args.smoke_tau
        run = train_c3(label, model, opt_cfg, stream, ctx, machine, out,
                       pool, grow_sink=grow_sink, smoke=args.smoke,
                       micro_override=args.microbatch)
        if run["status"] != "completed":
            raise RuntimeError(f"the {args.arm} arm {run['status']}")
        scores = score_arm(model, run, eval_windows, ctx, label)
        diagnostics = diagnostics_for_snapshots(model, run, diag_windows, ctx,
                                                pool, machine, label)
        heads_hist = growth_snapshot(model)["heads_histogram"] \
            if args.arm == "gdn2_growth" else None
        receipt = dict(
            probe="gdn2_growth_c3", arm=args.arm, seed=args.seed, context=ctx,
            tau_sat=tau, machine=machine,
            parameters=dict(
                total=sum(p.numel() for p in model.parameters()),
                delta_vs_registered=None),
            state_columns=config["state_columns"],
            exposure_bytes=run["seen"], exposure_steps=run["step"],
            constant_steps=run["constant_steps"],
            scores=scores,
            walls=dict(train_seconds=round(run["elapsed"], 1)),
            spectra_pool=pool.stats(),
            growth_telemetry=dict(
                events_total=len(run["wake_receipts"]),
                wake_up_receipts=run["wake_receipts"],
                final_heads_histogram=heads_hist,
                per_step_events=[dict(step=r["step"], growth=r["growth"])
                                 for r in run["rows"] if r["growth"]],
                probe_samples=[dict(step=r["step"], probe=r["probe"])
                               for r in run["rows"] if r.get("probe")])
            if args.arm == "gdn2_growth" else None,
            emender_native_diagnostics=None,
            saturation_diagnostics=diagnostics,
            config_sha256=matched_arms.config_sha256(config), config=config,
            session=matched_arms.session_meta(card),
            trajectory=matched_arms.trajectory_sample(
                run["rows"], keep_first=5, keep_last=5, stride=25))
        # the parameter delta vs the registered arm (first-class column)
        registered = 149_855_670 if machine == "gdn2" else 149_981_808
        receipt["parameters"]["delta_vs_registered"] = \
            receipt["parameters"]["total"] - registered
        path = (RECEIPTS / f"gdn2_growth_c3_{label}.json")
        atomic_json(path, receipt)
        print(json.dumps({k: receipt[k] for k in
                          ("arm", "context", "exposure_bytes",
                           "exposure_steps")}, indent=1), flush=True)
        if receipt["scores"]["primary"]:
            print(f"  {label} PRIMARY (post-cooldown): "
                  f"{receipt['scores']['primary']['eval_nll']:.4f}",
                  flush=True)
        return receipt
    finally:
        pool.close()


def cmd_lock(args) -> dict:
    """The supervisor's calibrated-τ branch (the 2026-11-22 ruling on the
    calibration's escalation receipt): the registered τ_sat = 96 is
    UNREACHABLE on real text at both scanned horizons, so the locked
    trigger is τ_cal = the trained fixed machine's own per-layer
    smoothed-PR ceiling at the locked horizon (8192) -- growth fires
    exactly when a boundary is MORE saturated than anything the fixed
    machine exhibits.  Committed BEFORE any growth arm runs (the §5
    calibration branch's own rule).  CPU-only: it reads the committed
    calibration receipts and writes the lock."""
    cal_path = RECEIPTS / "gdn2_growth_c3_calibration_8192.json"
    cal = json.loads(cal_path.read_text())
    if cal["tau_reachable"]:
        raise RuntimeError("the 8192 calibration says 96 IS reachable; the "
                           "calibrated branch does not apply")
    ceiling = max(cal["pr"]["post_cooldown"]["per_layer_smoothed_max"]
                  .values())
    argmax = [k for k, v in cal["pr"]["post_cooldown"]
              ["per_layer_smoothed_max"].items() if v == ceiling]
    lock = dict(
        locked=True, context=8192,
        tau_sat=ceiling,
        tau_registered=TAU_SAT,
        tau_branch=dict(
            rule="the supervisor's calibrated branch (the escalation ruling "
                 "on the committed calibration receipts): tau_cal = the "
                 "trained fixed machine's own per-layer smoothed-PR ceiling "
                 "at the locked horizon, measured on the post-cooldown "
                 "(PRIMARY-scored) trained state",
            value=ceiling, argmax_layer=argmax,
            semantics="growth fires exactly when a boundary is MORE "
                      "saturated than anything the fixed machine exhibits",
            hazard="the K8 idle-store hazard stated verbatim: PR ~20 of 128 "
                   "is the fixed machine's own regime ceiling, NOT absolute "
                   "capacity; the growth arm tests growth-past-the-exhibited-"
                   "regime, not growth at capacity.  The registered-96 "
                   "informative-null branch's verdict is delivered by the "
                   "reachability receipts themselves (PR max ~25 vs 96 at "
                   "both horizons) and is receipted permanently there",
            event_receipts="every firing event receipts its PR value, the "
                          "per-head PR at the fire boundary, the window "
                          "position (boundary), the post-event first-"
                          "boundary per-head PR, and the mean PR drop at "
                          "the event",
            reachability=dict(
                horizon_8192=dict(
                    receipt="receipts/gdn2_growth_c3_calibration_8192.json",
                    tau_reachable=False, pr_mean=cal["pr"]["post_cooldown"]
                    ["mean"], pr_max=cal["pr"]["post_cooldown"]["max"]),
                horizon_16384=dict(
                    receipt="receipts/"
                            "gdn2_growth_c3_calibration_16384.json",
                    tau_reachable=json.loads((RECEIPTS /
                        "gdn2_growth_c3_calibration_16384.json").read_text())
                    ["tau_reachable"])),
            fixed_arm_primary_reference=cal["scores"]["primary"]["eval_nll"]),
        horizon_rule="8192 LOCKED per the ruling (the shortest scanned "
                     "horizon; same reachability verdict; the emender fold "
                     "arithmetic is decisive: 4x deeper, not 8x)",
        protocol=dict(exposure_targets=EXPOSURE_TARGETS,
                      windows_per_step=windows_per_step(8192),
                      landing_targets=175 * windows_per_step(8192) * 8192,
                      constant_steps_expected=175,
                      cooldown_steps=COOLDOWN_STEPS,
                      cooldown_fraction=round(COOLDOWN_STEPS / 175, 4),
                      segment=SEGMENT, head_cap=HEAD_CAP, smooth=SMOOTH,
                      seed=SEED, eval_windows=EVAL_WINDOWS,
                      microbatch=dict(gdn2=microbatch_for(8192, "gdn2"),
                                      emender=microbatch_for(8192, "emender")),
                      stream="the registered flagship stream's documents, "
                             "full 8192-byte windows from eligible "
                             "documents only (byte_length >= 8193), "
                             "document-bounded, split-disjoint",
                      optimizer="plain AdamW at the registered gdn2 "
                                "nomination's transferred settings, all "
                                "four arms (the K6/K8/side-by-side "
                                "convention; identical numbers on both "
                                "machines)"),
        predictions=dict(PREDICTIONS, tau_note=
            "P-G2's 'trigger fired in-regime' clause reads at the CALIBRATED "
            "tau (the registered 96's informative-null verdict is the "
            "reachability receipt); P-G3's adjudication is PENDING the "
            "emender arm's landing (the ~36h matched-exposure long pole, "
            "detached per the ruling)"),
        eligible_documents=cal["eligible_documents"],
        bindings=regime_bindings(), git=matched_arms.git_state())
    atomic_json(RECEIPTS / "gdn2_growth_c3_regime.json", lock)
    print(json.dumps(dict(locked=True, context=8192, tau_cal=ceiling,
                          argmax_layer=argmax), indent=1), flush=True)
    return lock


def cmd_assemble(args) -> dict:
    """The four-arm table + the P-G1/P-G2/P-G3 adjudication (the margins read
    against the pre-stated 0.075 endpoint band, post-cooldown PRIMARY)."""
    lock = json.loads((RECEIPTS / "gdn2_growth_c3_regime.json").read_text())
    arms = {}
    for arm in ARMS:
        path = RECEIPTS / f"gdn2_growth_c3_{arm}.json"
        if not path.exists():
            arms[arm] = dict(missing=True, receipt=str(path))
            continue
        r = json.loads(path.read_text())
        arms[arm] = dict(
            primary=r["scores"]["primary"]["eval_nll"]
            if r["scores"]["primary"] else None,
            secondary=r["scores"]["secondary"]["eval_nll"]
            if r["scores"]["secondary"] else None,
            constant_final=[c["eval_nll"] for c in
                            r["scores"]["constant_curve"]][-1]
            if r["scores"]["constant_curve"] else None,
            parameters=r["parameters"]["total"],
            delta_vs_registered=r["parameters"]["delta_vs_registered"],
            exposure_bytes=r["exposure_bytes"],
            pr_mean=r["saturation_diagnostics"].get(
                "post-cooldown", {}).get("pr", {}).get("mean")
            if r.get("saturation_diagnostics", {}).get("post-cooldown")
            else None,
            iota_mean=None,
            growth_events=(r["growth_telemetry"]["events_total"]
                           if r.get("growth_telemetry") else None),
            state=r["state_columns"]["per_layer_state"])
    def primary(arm):
        v = arms[arm].get("primary")
        return v
    adjudication = dict(
        P_G1=dict(
            prediction=PREDICTIONS["P_G1"],
            receipted_by="the calibration scan (the regime lock's tau "
                         "reachability)",
            tau_reachable_at_locked_horizon=lock.get("scanned", {}).get(
                str(lock.get("context")), {}).get("tau_reachable")
            if lock.get("locked") else None),
        P_G2=dict(prediction=PREDICTIONS["P_G2"]),
        P_G3=dict(prediction=PREDICTIONS["P_G3"]),
        fixed_8=dict(prediction=PREDICTIONS["fixed_8"]),
    )
    complete = all(not arms[a].get("missing") and primary(a) is not None
                   for a in ARMS)
    if complete:
        m_growth = primary("gdn2_fixed") - primary("gdn2_growth")
        m_slots = primary("gdn2_growth") - primary("emender_growable")
        m_size = primary("gdn2_growth") - primary("gdn2_fixed_8")
        fired = bool(arms["gdn2_growth"].get("growth_events"))
        adjudication["P_G2"].update(
            margin_fixed_minus_growth=m_growth,
            trigger_fired=fired,
            verdict=("SUPPORTED" if (m_growth > ENDPOINT_BAND and fired)
                     else ("REFUTED (never fired): the informative null -- "
                           "growth degenerates to fixed in-regime"
                           if not fired else
                           "REFUTED (no gain outside the band): the "
                           "informative null")))
        adjudication["P_G3"].update(
            margin_growth_minus_emender=m_slots,
            verdict=("SUPPORTED" if m_slots > ENDPOINT_BAND else
                     "REFUTED: addressable slots do not beat growth-at-"
                     "matched-state at this scale (within the band or "
                     "GDN2+growth ahead)"))
        adjudication["fixed_8"].update(
            margin_growth_minus_fixed8=m_size,
            verdict="reported; the born-large vs grown-large control, no "
                    "pre-stated direction")
        table = {a: dict(primary=primary(a),
                         secondary=arms[a].get("secondary"),
                         constant_final=arms[a].get("constant_final"),
                         parameters=arms[a]["parameters"],
                         delta_vs_registered=arms[a]["delta_vs_registered"],
                         growth_events=arms[a].get("growth_events"),
                         exposure_bytes=arms[a]["exposure_bytes"],
                         state=arms[a]["state"]) for a in ARMS}
    else:
        table = arms
    summary = dict(
        probe="gdn2_growth_c3 assemble", regime_lock=lock,
        band_nats_per_byte=ENDPOINT_BAND,
        arms=table, adjudication=adjudication,
        complete=complete,
        bindings=regime_bindings(), git=matched_arms.git_state())
    atomic_json(RECEIPTS / "gdn2_growth_c3_summary.json", summary)
    print(json.dumps(dict(complete=complete,
                         margins={k: adjudication[k].get(
                             "margin_fixed_minus_growth",
                             adjudication[k].get("margin_growth_minus_emender"))
                             for k in ("P_G2", "P_G3")} if complete else None,
                         verdicts={k: adjudication[k].get("verdict")
                                   for k in ("P_G2", "P_G3")}
                         if complete else None), indent=1, default=str),
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("kshapes", help="kernel/shape receipts at the "
                                       "regime's horizons")
    p.add_argument("--out", type=Path,
                   default=RECEIPTS / "gdn2_growth_c3_kshapes.json")

    sub.add_parser("calibrate", help="the tau_sat calibration scan on both "
                                     "horizons + the regime lock")

    sub.add_parser("lock", help="the supervisor's calibrated-tau lock "
                                "(reads the committed calibration receipts; "
                                "CPU-only)")

    p = sub.add_parser("arm", help="one C3 arm at the locked regime")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke", action="store_true",
                   help="the machinery shakedown (tiny exposure; "
                        "receipted, never scored)")
    p.add_argument("--microbatch", type=int, default=None,
                   help="memory-knob override (grad accumulation is exact; "
                        "the growth arm runs checkpoint-free, so its "
                        "activation footprint is the honest price and the "
                        "override is receipted in the arm's config)")
    p.add_argument("--smoke-tau", type=float, default=40.0,
                   help="SMOKE-ONLY forced trigger threshold (the arms "
                        "always run the locked tau)")

    p = sub.add_parser("assemble", help="the four-arm table + the "
                                        "adjudication")

    args = parser.parse_args()
    if args.command == "kshapes":
        receipt = cmd_kshapes(args)
        atomic_json(args.out, receipt)
        print(json.dumps(dict(verdict=receipt["verdict"]), indent=1))
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "lock":
        cmd_lock(args)
    elif args.command == "arm":
        cmd_arm(args)
    elif args.command == "assemble":
        cmd_assemble(args)


if __name__ == "__main__":
    main()
