"""K24 -- THE LONG-WINDOW PROBE: the horizon hypothesis's first test, at
probe scale (does memory become load-bearing when the regime demands
state?).

THE QUESTION (``PREREGISTRATION_K24_LONG_WINDOW.md``, committed BEFORE any
arm): the uniformity finding (4-5 load-bearing layers of 24, the slot reads
decorative even inside the K22 hybrid) was measured at the banked probe
window length (2048 bytes, the 20,022,866-target matched-exposure landing).
If the state-demand lever -- WINDOW LENGTH -- is what the slots are FOR,
the load-bearing mask should light up as windows grow.  This lane is that
first test: a window-length bracket {1x, 2x, 4x} over the machines
{ours-composed, gdn2} plus {4x} over {hybrid} -- SEVEN arms, one exposure
each.

THE ARMS (seed 3401, matched 20,000,000-target stopping, the corrected
cooldown-primary scoring):

``ours_1x``    THE ANCHOR: the committed K15 composition machine (the banked
               1.3811104467936925 row's own build/recipe/protocol) re-run
               through THIS lane's loop at the banked window shape -- it
               must reproduce the banked row BIT FOR BIT (step-1
               5.782775391046648, every training-log row, the constant-180
               1.433674931526184, the post-cooldown landing
               1.3811104467936925 and the EMA-180
               1.424389226096017).  Any mismatch STOPS the lane.
``gdn2_1x``   THE ANCHOR: the committed GDN2 arm (the banked
               1.3521734476089478 row's own build/recipe) through the same
               loop -- step-1 5.948033220866492, the constant-180
               1.3950541530336653, the post-cooldown
               1.3521734476089478, the EMA-180 1.3803383282252721, all bit
               for bit.  Any mismatch STOPS the lane.
``ours_2x``/``ours_4x``    the same committed machine at the longer windows.
``gdn2_2x``/``gdn2_4x``    the same committed GDN2 at the longer windows.
``hybrid_4x``  the K22 hybrid engine (GDN2 + the composed slot mixers at
               {6, 12, 18} through the zero-born read gates) at 4x, born as
               GDN2 at the 4x batch (receipted vs a pure-GDN2 twin), the
               gates learn to open -- do they open FURTHER when state is
               demanded?

THE WINDOW-LENGTH LEVER (the standing discipline, never violated): the
longer windows are built from the corpus pipeline's own boundary-respecting
windowing (``lattice.Stream`` verbatim, constructed with a longer
``context``): every window is a slice of exactly ONE document, windows
TRUNCATE at the boundary, the document-disjoint splits stay intact, every
document slice is SHA-256-verified at consumption.  No engine is modified
to force anything -- if a machine could not honor the boundaries at longer
windows the lane would STOP, not force it.

THE STEP/WINDOW ARITHMETIC (the probe's own step shape, adjusted so every
arm eats the SAME bytes; matched exposure is the invariant, stated per the
registration): the banked 1x protocol is 64 windows/step x 2048 bytes with
microbatch 8 (16,384 tokens per microbatch) landing at step 180 /
20,022,866 targets.  The longer arms hold (a) the ~111K-targets-per-step
shape and (b) the ~180-step landing, by scaling windows/step DOWN as the
context grows, and hold the microbatch's TOKEN count at the banked 16,384
(the memory-matched rule -- the banked ours arm's peak allocation is 27.9
GiB at 16,384 tokens/microbatch and the card is 48 GiB):

============  =======  ==========  =========  ===============  ==========
context       windows  microbatch  tokens/    constant-phase   exposure
(bytes)       /step    (windows)   microbatch landing (step)   (targets)
============  =======  ==========  =========  ===============  ==========
2048 (1x)     64       8           16,384     180              20,022,866
4096 (2x)     38       4           16,384     182              20,067,118
8192 (4x)     27       2           16,384     178              20,088,160
============  =======  ==========  =========  ===============  ==========

THE EFFECTIVE WINDOW-LENGTH DISTRIBUTION (receipted per arm, honestly: the
lever is only as strong as what was actually achieved -- the corpus's
median document is ~1,850 bytes, SHORTER than one banked window):

=====  ========  ========  ========  ===========  =====================
mult  mean      p50       p90       full-length  consumed windows
      (bytes)   (bytes)   (bytes)   fraction     (to the 20M landing)
=====  ========  ========  ========  ===========  =====================
1x    1738.1    2048      2048      0.718         11,520
2x    2901.5    4096      4096      0.530         6,916
4x    4179.8    3042      8192      0.323         4,806
=====  ========  ========  ========  ===========  =====================

The 4x arm's ACHIEVED mean window is 2.4x the banked mean (not 4x); the
2x arm's is 1.67x.  Every arm's receipt states its own achieved
distribution; the curve is read against what was achieved.

THE EVAL RULE: the banked calibration eval is L.evaluate's own -- the
first 24 calibration windows in split order, keeping the full ones (7
windows / 14,336 targets at 1x).  The K24 rule for the longer arms is the
same rule's intent at the arm's own window: THE FIRST 7 FULL WINDOWS of
the arm's own context in calibration order -- at 1x the two rules pick
the identical window set (receipted from the manifest at registration
time and asserted by the lane's tests), at 2x 28,672 targets, at 4x
57,344 targets.  The 1x cells evaluate through ``L.evaluate`` VERBATIM.
Every arm is additionally CROSS-EVALUATED at all three window lengths on
its post-cooldown final state (the conditioning-vs-regime decomposition).

THE INSTRUMENTS (the decisive duties, per arm): THE DECORATIVE SWEEP AS
THE HEADLINE (the load-bearing mask per layer vs window length -- the
curve IS the result; the 0.02 dNLL bar, the banked 1x regime's own 4-5
layers); the C5 instrument (the composed walk's effective rows /
occupancy / identity share / op mix / gate telemetry); GDN2's own state
saturation telemetry (the retention exp(g), the implied memory horizon,
the erase/write gates); the hybrid's gate trajectories on the slot layers
at 4x (vs the banked 1x hybrid's own row); the at-birth receipts for the
hybrid at 4x (born as GDN2 vs a pure twin on the same 4x microbatch, the
open signal, the inert birth, the store fills).

PRE-STATE (directional, the hypothesis under test, with the refutation
clause): L-1 (the horizon hypothesis) -- the slot layers' load-bearing
count RISES with window length (more layers clear the 0.02 bar at 4x than
at 1x, and/or the occupancy/effective rows rise on the slot machines);
L-2 -- the quality gap (ours - gdn2) SHRINKS with window length.  REFUTATION,
receipted plainly: if the mask stays dark and the gap stays flat at 4x,
the horizon hypothesis is dead at probe scale.  No band-worship: every
delta is stated against the 0.0119 winner-seed band.

PROVENANCE (the lane's own rules): the engines are committed and
read-only -- ``velocity_ladder._build_k10`` + ``k15_composition`` (ours),
``matched_arms.build_gdn2`` (GDN2), ``k22_hybrid`` (the hybrid engine),
``bakeoff_optimizers.HybridMuonX45``, ``muon_control`` /
``k11b_probe_channel_gates`` routing splits, ``protocol_correction``'s
corrected-scoring conventions and ``lattice``'s stream -- all imported
VERBATIM, never modified.  This file only WRAPS: the one new thing is the
window-length threading (the longer-context stream construction, the
context-aware grouping, the longer-window instruments) and the K24
training loop -- a character-faithful copy of
``protocol_correction._train_corrected`` whose ONLY differences are the
threaded (context, windows_per_step) pair, pinned at 1x by the anchors
themselves (an anchor mismatch is a STOP, not a bug fix).

Usage (see ``run_k24_long_window_leased.sh`` for the detached leased
driver, cards 2/3, one card at a time, commit per arm):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \\
        arm --arm ours_4x --seed 3401 --output k24_runs/ours_4x_seed3401
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

# The probe subprocess discipline (configure_production fail-closes on a
# missing allocator knob if these are not set before CUDA initialization).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
    import (COOLDOWN_STEPS, CORRECTED_SCORING, EMA_DECAY,
            ENDPOINT_VARIANCE_BAND, GRAD_CLIP, WINNER_SEED_DELTA,
            cooldown_factor)
from experiments.expressivity_tasks.x45_matrix_emender.\
    k15_probe_composition import PROBE_WINDOWS, _k15_walk_stats, \
    _tuned_winner
from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import composed_4x_banked
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    GDN2_BANKED_ADAMW)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k24_runs"

# ---- the window-length bracket (the registration's own numbers) ----
WINDOW_MULTS = (1, 2, 4)
WINDOW_CONTEXT = {1: 2048, 2: 4096, 4: 8192}     # bytes (lattice.CONTEXT x mult)
WINDOWS_PER_STEP = {1: 64, 2: 38, 4: 27}         # the ~111K-targets/step shape
MICROBATCH = {1: 8, 2: 4, 4: 2}                  # windows per microbatch
TOKENS_PER_MICROBATCH = 16384                    # the memory-matched rule
EXPOSURE_TARGETS = 20_000_000                    # the banked stopping rule
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
EVAL_WINDOWS = 7        # the first 7 FULL windows (the banked 1x set's own)
PRIMARY_SEED = 3401
LOAD_BEARING_BAR = 0.02  # the uniformity finding's own bar

# ---- the banked rows this lane cites (never re-run, never rewritten) ----
# OURS: the committed K15 composition arm's own landed rows
# (k15_runs/composition_seed3401, the 1.3811104467936925 machine).
OURS_1X_RUN = ROOT / "k15_runs" / "composition_seed3401"
# GDN2: the committed arm's own corrected-scoring row
# (protocol_correction_runs/gdn2_speedrun_rescored, the 1.3522 machine).
GDN2_1X_RUN = ROOT / "protocol_correction_runs" / "gdn2_speedrun_rescored"
# THE HYBRID's banked 1x row (the K22 lane's own; cited, never re-run).
HYBRID_1X_RUN = ROOT / "k22_runs" / "hybrid_seed3401"
BANDS = dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
             winner_seed_delta=WINNER_SEED_DELTA)

ARMS = {
    "ours_1x": dict(machine="ours", mult=1, role="anchor",
                    note="THE ANCHOR: the committed K15 composition machine "
                         "re-run at the banked window shape through THIS "
                         "lane's loop -- must reproduce the banked row bit "
                         "for bit (any mismatch STOPS the lane)"),
    "gdn2_1x": dict(machine="gdn2", mult=1, role="anchor",
                    note="THE ANCHOR: the committed GDN2 arm re-run at the "
                         "banked window shape through THIS lane's loop -- "
                         "must reproduce the banked 1.3522 row bit for bit "
                         "(any mismatch STOPS the lane)"),
    "ours_2x": dict(machine="ours", mult=2, role="long",
                    note="the committed composed machine at the 2x windows"),
    "gdn2_2x": dict(machine="gdn2", mult=2, role="long",
                    note="the committed GDN2 arm at the 2x windows"),
    "ours_4x": dict(machine="ours", mult=4, role="long",
                    note="the committed composed machine at the 4x windows"),
    "gdn2_4x": dict(machine="gdn2", mult=4, role="long",
                    note="the committed GDN2 arm at the 4x windows"),
    "hybrid_4x": dict(machine="hybrid", mult=4, role="long",
                      note="the K22 hybrid engine at the 4x windows: born "
                           "as GDN2 at the 4x batch (receipted vs a pure "
                           "twin), the gates learn to open -- do they open "
                           "FURTHER when state is demanded?"),
}

RULES = dict(
    L1=dict(
        text="L-1 (the horizon hypothesis): the slot layers' load-bearing "
             "count RISES with window length -- the decorative sweep lights "
             "up (more layers clear the 0.02 dNLL bar at 4x than at 1x; the "
             "banked 1x regime's own count is 4 of 24 on ours, 0 of 3 slot "
             "layers on the hybrid), and/or the occupancy/effective rows "
             "rise on the slot machines.",
        bar=LOAD_BEARING_BAR),
    L2=dict(
        text="L-2: the quality gap (ours - gdn2) SHRINKS with window length "
             "(the dense state saturates as sequences grow; the addressable "
             "store does not).  Read on the own-window post-cooldown rows at "
             "each mult, with the fixed-1x-eval cross-grid as the "
             "decomposition."),
    REFUTATION=dict(
        text="REFUTATION, receipted plainly: if the mask stays dark (the "
             "load-bearing count does not rise at 4x) and the gap stays "
             "flat at 4x (inside the 0.0119 seed band of the 1x gap), the "
             "horizon hypothesis is dead at probe scale -- the board run's "
             "horizon rationale goes with it, and the remaining board value "
             "is scale/exposure alone."),
    band_discipline=dict(
        text="no band-worship: every delta is stated against the 0.0119 "
             "winner-seed band (and the 0.075 endpoint-variance band "
             "cross-machine); a delta inside the seed band is PARITY, "
             "phrased as parity."),
    anchor=dict(
        text="the anchor convention: the 1x cells MUST reproduce the banked "
             "rows bit for bit FIRST (ours 1.3811104467936925, GDN2 "
             "1.3521734476089478; the full training-log trajectory, the "
             "constant-180, the EMA-180 and the landing, every digit).  "
             "ANY mismatch STOPS the lane and escalates -- before any long "
             "window run."),
    boundaries=dict(
        text="the standing discipline: NEVER pack across document "
             "boundaries; windows TRUNCATE at the boundary; the "
             "document-disjoint splits stay intact; every document slice "
             "SHA-256-verified at consumption.  The longer windows are the "
             "corpus pipeline's own boundary-respecting windowing (the "
             "lattice.Stream constructor's own context parameter), "
             "hash-bound to the flagship manifest.  An engine that cannot "
             "honor the boundaries at longer windows is a STOP, never a "
             "forced fit."),
)


# ------------------------------------------------------------- plumbing


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def k24_bindings() -> dict:
    """Hash-bind every authority and engine this lane runs under."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    return dict(
        registration=str(ROOT / "PREREGISTRATION_K24_LONG_WINDOW.md"),
        registration_sha256=digest(ROOT / "PREREGISTRATION_K24_LONG_WINDOW.md"),
        k24_long_window_sha256=digest(ROOT / "k24_long_window.py"),
        k24_gate_receipt_sha256=digest(ROOT / "k24_gate_receipt.py"),
        k15_composition_sha256=digest(ROOT / "k15_composition.py"),
        k15_probe_sha256=digest(ROOT / "k15_probe_composition.py"),
        k22_hybrid_sha256=digest(ROOT / "k22_hybrid.py"),
        k22_probe_sha256=digest(ROOT / "k22_probe_hybrid.py"),
        matched_arms_sha256=digest(ROOT / "matched_arms.py"),
        muon_control_sha256=digest(ROOT / "muon_control.py"),
        protocol_correction_sha256=digest(ROOT / "protocol_correction.py"),
        external_gdn2_sha256=digest(REPO / "ndm/models/external_gdn2.py"),
        lattice_registration_sha256=identity["lattice_sha256"],
        flagship_manifest_sha256=identity["manifest_sha256"],
        banked_ours_row=dict(
            run_dir=str(OURS_1X_RUN),
            start_sha256=digest(OURS_1X_RUN / "start.json"),
            result_sha256=digest(OURS_1X_RUN / "result.json"),
            training_log_sha256=digest(OURS_1X_RUN / "training.jsonl")),
        banked_gdn2_row=dict(
            run_dir=str(GDN2_1X_RUN),
            start_sha256=digest(GDN2_1X_RUN / "start.json"),
            result_sha256=digest(GDN2_1X_RUN / "result.json"),
            training_log_sha256=digest(GDN2_1X_RUN / "training.jsonl")),
        banked_hybrid_1x_row=dict(
            run_dir=str(HYBRID_1X_RUN),
            result_sha256=digest(HYBRID_1X_RUN / "result.json")),
    )


def _load_run(path: Path) -> dict:
    """A banked run's own rows, loaded from its committed receipts (the
    constants are pinned by the lane's tests; the loader never restates
    them)."""
    result = json.loads((path / "result.json").read_text())
    rows = [json.loads(line) for line in
            (path / "training.jsonl").read_text().splitlines()]
    return dict(result=result, rows=rows)


def ours_1x_banked() -> dict:
    run = _load_run(OURS_1X_RUN)
    r = run["result"]
    return dict(post_cooldown=r["final_eval_nll_post_cooldown"],
                constant_lr=r["final_eval_nll_constant_lr"],
                ema180=r["ema_eval_nll"], step1=r["step1_train_loss"],
                exposure_bytes=r["exposure_bytes"],
                total_bytes=r["total_bytes"], steps=r["exposure_steps"],
                rows=run["rows"], run_dir=str(OURS_1X_RUN))


def gdn2_1x_banked() -> dict:
    run = _load_run(GDN2_1X_RUN)
    r = run["result"]
    return dict(post_cooldown=r["final_eval_nll_post_cooldown"],
                constant_lr=r["final_eval_nll_constant_lr"],
                ema180=r["ema_eval_nll"], step1=r["step1_train_loss"],
                exposure_bytes=r["exposure_bytes"],
                total_bytes=r["total_bytes"], steps=r["exposure_steps"],
                rows=run["rows"], run_dir=str(GDN2_1X_RUN))


def hybrid_1x_banked() -> dict:
    """The K22 hybrid's banked 1x row (cited in the curve's hybrid cell)."""
    r = json.loads((HYBRID_1X_RUN / "result.json").read_text())
    sweep = r.get("slot_read_mask_sweep_post_cooldown") or {}
    gates = r.get("gate_trajectory") or {}
    diag = r.get("diagnostics_post_cooldown") or {}
    return dict(post_cooldown=r["final_eval_nll_post_cooldown"],
                constant_lr=r["final_eval_nll_constant_lr"],
                ema180=r["ema_eval_nll"], parameters=r["parameters"],
                gates_final=gates.get("final"),
                slot_read_mask_sweep=dict(
                    delta_per_slot_layer=sweep.get("delta_per_slot_layer"),
                    all_slots_delta=sweep.get("all_slots_delta"),
                    slot_layers_ge_bar=sweep.get("slot_layers_ge_bar")),
                mechanism=dict(
                    effective_rows=diag.get("effective_read_rows"),
                    occupancy=(diag.get("occupancy") or {}).get(
                        "mean_live_slots"),
                    identity_share=diag.get("identity_rewrite_share"),
                    op_mix=diag.get("op_rate_per_token_lane0")),
                run_dir=str(HYBRID_1X_RUN))


# ------------------------------------------- the window-length arithmetic


_LENGTH_CACHE: dict = {}


def _manifest_documents(manifest: Path):
    opener = gzip.open if manifest.suffix == ".gz" else open
    with opener(manifest, "rt") as f:
        for line in f:
            d = json.loads(line)
            yield dict(split=d["split"], length=int(d["byte_length"]))


def window_lengths(manifest: Path, split: str, context: int) -> list:
    """The boundary-respecting window-length sequence for a split at a
    context, mirroring ``lattice.Stream.windows`` exactly (starts at
    document-relative multiples of the context; every window truncates at
    the boundary; a document shorter than context+1 contributes its one
    short window; nothing is packed, nothing crosses)."""
    key = (str(manifest), split, context)
    if key in _LENGTH_CACHE:
        return _LENGTH_CACHE[key]
    manifest = Path(manifest)
    lengths = []
    for d in _manifest_documents(manifest):
        if d["split"] != split:
            continue
        remaining = d["length"] - 1
        start = 0
        while start < remaining:
            lengths.append(min(context, remaining - start))
            start += context
    _LENGTH_CACHE[key] = lengths
    return lengths


def window_arithmetic(manifest: Path, mult: int,
                      windows_per_step: int = None,
                      targets: int = EXPOSURE_TARGETS) -> dict:
    """The per-arm step/window arithmetic receipt, computed from the
    manifest (never hardcoded): the consumed-prefix effective window-length
    distribution (mean/p50/p90, the full-length fraction), the windows
    consumed to the matched-exposure landing, the landing step and the
    landing exposure.  The training loop consumes whole steps, so the
    landing exposure is the sum over the landing step's whole window
    group -- exactly as the banked 1x protocol lands (11,520 windows,
    20,022,866 targets)."""
    import numpy as np
    context = WINDOW_CONTEXT[mult]
    windows_per_step = (WINDOWS_PER_STEP[mult] if windows_per_step is None
                        else windows_per_step)
    lengths = window_lengths(manifest, "train", context)

    def _crossing(step_lengths, step_windows_per_step, step_targets):
        """(crossing window count, landing step, group_end) for a
        windows-per-step shape: the training loop consumes whole steps, so
        the landing is the first step boundary at or after the crossing
        window -- exactly how the banked 1x protocol lands (11,520
        windows, 20,022,866 targets)."""
        seen = 0
        crossing = 0
        for length in step_lengths:
            seen += length
            crossing += 1
            if seen >= step_targets:
                break
        landing = -(-crossing // step_windows_per_step)
        return crossing, landing, landing * step_windows_per_step

    crossing, landing_step, group_end = _crossing(lengths, windows_per_step,
                                                  targets)
    consumed = lengths[:group_end]
    arr = np.array(consumed)
    calibration_full = sum(
        1 for w in window_lengths(manifest, "calibration", context)
        if w == context)
    # the 1x consumed prefix's own mean (the lever's honest scale)
    base_lengths = window_lengths(manifest, "train", WINDOW_CONTEXT[1])
    _c, base_landing, base_end = _crossing(base_lengths,
                                           WINDOWS_PER_STEP[1], targets)
    base_mean = float(np.mean(base_lengths[:base_end]))
    return dict(
        mult=mult, context=context, windows_per_step=windows_per_step,
        microbatch=MICROBATCH[mult],
        tokens_per_microbatch=MICROBATCH[mult] * context,
        microbatch_groups_per_step=-(-windows_per_step
                                     // MICROBATCH[mult]),
        landing_step=int(landing_step),
        exposure_landing=int(arr.sum()),
        windows_consumed=int(group_end),
        effective_window_bytes=dict(
            mean=round(float(arr.mean()), 1),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            min=int(arr.min()), max=int(arr.max()),
            full_length_fraction=round(float((arr == context).mean()), 4),
            mean_over_1x_mean=round(float(arr.mean()) / base_mean, 3)),
        calibration_full_windows=calibration_full,
        note=("the boundary-respecting windowing's own distribution over "
              "the windows the arm actually consumes (mean/p50/p90 bytes, "
              "the full-length fraction); the lever is only as strong as "
              "what was actually achieved -- the corpus's median document "
              "is ~1,850 bytes, shorter than one banked window"))


def registration_window_receipt() -> dict:
    """The registration-time window arithmetic for all three mults (bound
    to the flagship manifest), committed with the registration."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    manifest = Path(identity["manifest"])
    return dict(probe="k24_window_arithmetic",
                manifest=str(manifest),
                manifest_sha256=identity["manifest_sha256"],
                exposure_targets=EXPOSURE_TARGETS,
                tokens_per_microbatch=TOKENS_PER_MICROBATCH,
                eval_rule=("the first 7 FULL windows of the arm's own "
                           "context in calibration order; at 1x this picks "
                           "the banked eval's own window set (L.evaluate's "
                           "first-24-kept-full rule yields the same 7)"),
                arms={str(mult): window_arithmetic(manifest, mult)
                      for mult in WINDOW_MULTS})


# ------------------------------------------------------------- the streams


def make_streams():
    """The three streams (one per window mult), the lattice.Stream
    constructor's own context parameter -- the boundary-respecting
    windowing VERBATIM, never a repacking."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    streams = {mult: L.Stream(identity["source"], identity["manifest"],
                             context=WINDOW_CONTEXT[mult])
               for mult in WINDOW_MULTS}
    L.assert_flagship_stream(streams[1])
    return streams, identity


def k24_probe_rows(stream, mult: int, windows: int):
    """The instrument's own probe rows: the first N FULL windows of the
    stream's own context in calibration order (at 1x this is
    ``_probe_windows``'s own set)."""
    context = WINDOW_CONTEXT[mult]
    rows = [row for row in stream.windows("calibration")
            if len(row[2]) == context]
    if len(rows) < windows:
        raise ValueError(f"calibration split has {len(rows)} full "
                          f"{context}-byte windows; the probe needs "
                          f"{windows}")
    return rows[:windows]


def evaluate_at(model, stream, mult: int, windows: int = EVAL_WINDOWS):
    """The arm's own-window eval: at 1x the banked ``L.evaluate`` VERBATIM
    (its first-24-kept-full rule); at the longer mults the same rule's
    intent at the arm's own context -- the first 7 FULL windows in
    calibration order (the set equality with the banked rule at 1x is
    receipted at registration and asserted by the tests)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
    if mult == 1:
        return L.evaluate(model, stream, "matrix_ops", "calibration")
    context = WINDOW_CONTEXT[mult]
    model.eval()
    total = 0.0
    targets = 0
    used = []
    with torch.no_grad():
        for row in k24_probe_rows(stream, mult, windows):
            x, y, _valid = next(grouped([row], 1, context))
            total += float(model.loss(x, y)) * y.numel()
            targets += int(y.numel())
            used.append([row[0], row[1]])
    if not targets:
        raise ValueError(f"no full {context}-byte windows in calibration")
    return dict(score=total / targets, nll_heldout=total / targets,
                eval_split="calibration", eval_windows=len(used),
                eval_targets=targets, units="nats/byte")


# ------------------------------------------------- the K24 training loop


def _train_corrected_k24(model, optimizer, peak_lr, warmup, microbatch,
                         stream, context, windows_per_step, targets, label,
                         log_path, checkpoint_steps, cooldown_steps,
                         ema_decay, banked_rows=None):
    """The lanes' armed loop, VERBATIM from
    ``protocol_correction._train_corrected`` with exactly TWO threaded
    differences -- the stream is consumed as ``batches("train",
    context=context, batch=windows_per_step)`` and the grouping pads to the
    arm's own ``context`` -- plus the anchor rows hook (the 1x anchor arms
    compare EVERY step's loss to the banked row bit for bit, failing
    closed at the first divergence).  At mult 1 the threaded values are
    the banked protocol's own (context 2048, 64 windows/step), so the loop
    IS the banked loop -- the anchors pin this by construction."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    batches = iter(stream.batches("train", context=context,
                                  batch=windows_per_step))
    elapsed = 0.0
    opt_seconds = []
    step = seen = 0
    rows = []
    snapshots = {}
    diverged = False
    last_metrics = None
    ema = {name: param.detach().float().clone()
           for name, param in model.named_parameters()
           if param.requires_grad}

    def lr_at(step0, cooldown_k=0):
        factor = min(1., (step0 + 1) / max(1, warmup))
        if cooldown_k:
            factor *= cooldown_factor(cooldown_k, cooldown_steps)
        return peak_lr * factor

    def do_step(cooldown_k):
        nonlocal elapsed, step, seen, last_metrics, diverged
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            batch_rows = next(batches)
        except StopIteration:
            raise RuntimeError("train split exhausted before the probe budget")
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step, cooldown_k)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for x, y, valid in grouped(batch_rows, microbatch, context):
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if diverged:
            elapsed += time.monotonic() - start
            return None
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                       error_if_nonfinite=True)
        torch.cuda.synchronize()
        opt_start = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        opt_seconds.append(time.monotonic() - opt_start)
        last_metrics = getattr(optimizer, "last_metrics", None)
        elapsed += time.monotonic() - start
        step += 1
        seen += n
        row = dict(step=step, active_seconds=elapsed, bytes=seen,
                   loss=loss_sum, time_ns=time.time_ns(),
                   phase="cooldown" if cooldown_k else "constant",
                   lr_factor=(lr / peak_lr))
        rows.append(row)
        if banked_rows is not None:
            # the anchor's own bar: EVERY step bit-identical to the banked
            # row's loss (fail closed at the FIRST divergence -- a
            # birth-parity conflict, escalate, not excuse)
            if step > len(banked_rows):
                raise ValueError(
                    f"ANCHOR DIVERGED: the run passed the banked "
                    f"trajectory's last step ({len(banked_rows)})")
            banked = banked_rows[step - 1]
            if banked["step"] != step or row["loss"] != banked["loss"]:
                raise ValueError(
                    f"ANCHOR DIVERGED at step {step}: loss {row['loss']!r} "
                    f"!= the banked row's {banked['loss']!r} -- the 1x cell "
                    f"does not reproduce the banked trajectory bit for bit "
                    f"(STOP and escalate; the anchor convention is "
                    f"non-negotiable)")
        return row

    log = Path(log_path).open("w", buffering=1) if log_path else None
    try:
        # ---- phase 1: the banked protocol verbatim (constant LR)
        while seen < targets:
            row = do_step(0)
            if row is None:
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            opt_seconds=opt_seconds, ema=None,
                            constant_steps=step, muon_metrics=last_metrics)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            with torch.no_grad():      # the EMA (constant phase only)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema[name].mul_(ema_decay).add_(
                            param.detach().float(), alpha=1 - ema_decay)
            if step in checkpoint_steps:
                snapshots[step] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s "
                      f"{seen/1e6:.1f}MB loss {row['loss']:.4f}", flush=True)
        constant_steps = step
        constant_elapsed = elapsed
        # ---- phase 2: the linear cooldown (the corrected scoring's own
        # extension; the same stream continues, matched across arms)
        for k in range(1, cooldown_steps + 1):
            row = do_step(k)
            if row is None:
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            opt_seconds=opt_seconds, ema=ema,
                            constant_steps=constant_steps,
                            muon_metrics=last_metrics)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            if step % 25 == 0 or k == cooldown_steps:
                print(f"  {label} cooldown {k}/{cooldown_steps} "
                      f"{elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {row['loss']:.4f}", flush=True)
        snapshots["final"] = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                constant_steps=constant_steps,
                constant_elapsed=constant_elapsed, rows=rows,
                snapshots=snapshots, opt_seconds=opt_seconds, ema=ema,
                muon_metrics=last_metrics)


# --------------------------------------------------- the machine builders


def build_machine(machine: str, seed: int):
    """The three committed machines, their banked builds VERBATIM.  Returns
    (model, optimizer, optimizer_config, routing_receipt, optimizer_block,
    summary)."""
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45
    if machine == "ours":
        from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
            import _build_k10
        from experiments.expressivity_tasks.x45_matrix_emender.\
            k11b_probe_channel_gates import (_routing_cross_check,
                                             split_parameters_gated)
        from experiments.expressivity_tasks.x45_matrix_emender.\
            k15_composition import convert_model_to_composition
        model, optimizer_config, summary = _build_k10(dict(_tuned_winner()),
                                                       seed)
        model = convert_model_to_composition(model)
        model = model.cuda().train()
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_gated(model, "p3")
        routing_receipt = _routing_cross_check(model, "p3", seed, False)
        peak_lr = 4 * optimizer_config["lr"]     # the banked winner lineage
        optimizer = HybridMuonX45(
            muon_params, adamw_params, lr=peak_lr,
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"])
        optimizer_block = dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            peak_lr=peak_lr, lr_multiplier=4,
            lr_convention="4x the machine's own tuned AdamW LR -- the banked "
                          "winner lineage (the 1.3811 row's own recipe "
                          "VERBATIM)",
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=optimizer_config["warmup_steps"],
            microbatch=optimizer_config["microbatch"], grad_clip=1.0,
            muon_params=f"{len(muon_names)} tensors",
            adamw_params=f"{len(adamw_names)} tensors")
        return model, optimizer, optimizer_config, routing_receipt, \
            optimizer_block, summary
    if machine == "gdn2":
        from experiments.expressivity_tasks.x45_matrix_emender.muon_control \
            import split_parameters_gdn2
        model, optimizer_config, summary = matched_arms.build_gdn2(seed)
        for key, want in GDN2_BANKED_ADAMW.items():
            actual = (tuple(optimizer_config[key]) if key == "betas"
                      else optimizer_config[key])
            if actual != want:
                raise ValueError(f"the GDN2 machine's config drifted on "
                                 f"{key}: {actual!r} != the banked {want!r}")
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_gdn2(model, gates_on_adamw=True)
        peak_lr = 5 * GDN2_BANKED_ADAMW["lr"]    # the banked 5x speedrun point
        optimizer = HybridMuonX45(
            muon_params, adamw_params, lr=peak_lr,
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"])
        optimizer_block = dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            peak_lr=peak_lr, lr_multiplier=5,
            lr_convention="5x GDN2's own banked AdamW LR -- the banked "
                          "1.3522 row's own recipe VERBATIM (option B, "
                          "gates on AdamW)",
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=optimizer_config["warmup_steps"],
            microbatch=optimizer_config["microbatch"], grad_clip=1.0,
            muon_params=f"{len(muon_names)} tensors",
            adamw_params=f"{len(adamw_names)} tensors")
        return model, optimizer, optimizer_config, None, optimizer_block, \
            summary
    if machine == "hybrid":
        from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
            import (SLOT_LAYERS, _routing_cross_check_k22,
                    convert_gdn2_to_hybrid, hybrid_slot_parameter_delta,
                    hybrid_slot_layers, split_parameters_k22)
        model, optimizer_config, summary = matched_arms.build_gdn2(seed)
        for key, want in GDN2_BANKED_ADAMW.items():
            actual = (tuple(optimizer_config[key]) if key == "betas"
                      else optimizer_config[key])
            if actual != want:
                raise ValueError(f"the GDN2 machine's config drifted on "
                                 f"{key}: {actual!r} != the banked {want!r}")
        width = int(model.model.width)
        model = convert_gdn2_to_hybrid(model)
        model = model.train()
        parameters_total = sum(p.numel() for p in model.parameters())
        slot_delta = hybrid_slot_parameter_delta(width, SLOT_LAYERS)
        if parameters_total != 149_855_670 + slot_delta:
            raise ValueError(f"the hybrid's parameters must be the banked "
                             f"149,855,670 + the registration's "
                             f"{slot_delta}; got {parameters_total}")
        if tuple(hybrid_slot_layers(model)) != tuple(SLOT_LAYERS):
            raise ValueError("the slot layers drifted from the "
                             "registration's")
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_k22(model, False)
        routing_receipt = _routing_cross_check_k22(model, seed, False)
        peak_lr = 5 * GDN2_BANKED_ADAMW["lr"]
        optimizer = HybridMuonX45(
            muon_params, adamw_params, lr=peak_lr,
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"])
        optimizer_block = dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            peak_lr=peak_lr, lr_multiplier=5,
            lr_convention="the banked GDN2 row's own 5x HybridMuonX45 "
                          "recipe VERBATIM (option B + the slot branch per "
                          "the K18 convention mirrored; the K22 hybrid "
                          "engine's own)",
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=optimizer_config["warmup_steps"],
            microbatch=optimizer_config["microbatch"], grad_clip=1.0,
            muon_params=f"{len(muon_names)} tensors",
            adamw_params=f"{len(adamw_names)} tensors")
        return model, optimizer, optimizer_config, routing_receipt, \
            optimizer_block, summary
    raise ValueError(f"unknown machine {machine!r}")


# --------------------------------------------- the walk-stats hook bodies


def _composed_hook(per_layer, gate_rows, layer_index, mixer):
    """The K15 composed walk-stats hook body VERBATIM (the composed
    machine's own instrument: the walk on the mixer's own inputs, the
    per-channel gate telemetry), collected at ANY token count."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    pol32 = (_round_f32(mixer.policy.tau_match),
             _round_f32(mixer.policy.tau_merge),
             _round_f32(mixer.policy.tau_pop),
             _round_f32(mixer.policy.tie_break),
             _round_f32(mixer.policy.read_scale))

    def hook(_module, args, _output):
        xin = args[0]
        batch, tokens, _ = xin.shape
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            payload, write_keys, query_keys, _scalar = _project_keyed(
                mixer, xin)
            payload, write_keys = payload.float(), write_keys.float()
            query_keys = query_keys.float()
            lam_vec, lam_inc = mixer.channel_decay(xin)
            drift = (lam_vec.float() - lam_inc.float().unsqueeze(-1)).abs()
            gate_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(mixer.b_proj(xin)).view(
                    batch, tokens, mixer.heads, mixer.value_dim)
            gw_dist = (1.0 - gate_vec.float()).abs()
            r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(mixer.g_proj(xin)).view(
                    batch, tokens, mixer.heads, mixer.value_dim)
            gr_dist = (r_vec.float() - 1.0).abs()
            lam_vec32 = lam_vec.float()
            gates32 = gate_vec.float()
            state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                mixer.value_dim, dtype=torch.float32,
                                device=xin.device)
            live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               dtype=torch.bool, device=xin.device)
            keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               mixer.key_dim, dtype=torch.float32,
                               device=xin.device)
            out = _k15_forward_launch(
                payload, write_keys, query_keys, lam_vec32, gates32,
                state, live, keys, pol32, mixer.rank,
                mixer.segment_tokens, True, collect=True, warps=8, stages=3)
        with torch.no_grad():
            stats, _schedule = _k15_walk_stats(
                out, mixer.rank, mixer, payload=payload,
                decay_vec=lam_vec32)
            per_layer.append((layer_index, stats))
            gate_rows.append(dict(
                layer=layer_index,
                decay_gate=dict(
                    mean_distance_from_registered_scalar=float(drift.mean()),
                    max=float(drift.max()),
                    lam_vec_mean=float(lam_vec32.mean()),
                    lam_vec_min=float(lam_vec32.min()),
                    lam_vec_max=float(lam_vec32.max())),
                write_gate=dict(
                    mean_distance_from_1=float(gw_dist.float().mean()),
                    max=float(gw_dist.float().max())),
                read_gate=dict(
                    mean_distance_from_1=float(gr_dist.float().mean()),
                    max=float(gr_dist.float().max()))))
    return hook


def _slot_hook(per_layer, gate_rows, conv_rows, layer_index, mixer):
    """The K22 slot-mixer hook body VERBATIM (the composed walk on the
    hybrid's slot mixers: the K18 identity-born conv on the payload, the
    walk stats, the p3 gate telemetry, the conv kernel telemetry)."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv1d_causal
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    pol32 = (_round_f32(mixer.policy.tau_match),
             _round_f32(mixer.policy.tau_merge),
             _round_f32(mixer.policy.tau_pop),
             _round_f32(mixer.policy.tie_break),
             _round_f32(mixer.policy.read_scale))

    def hook(_module, args, _output):
        xin = args[0]
        batch, tokens, _ = xin.shape
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _payload, write_keys, query_keys, _scalar = _project_keyed(
                mixer, xin)
            payload = short_conv1d_causal(
                mixer.value(xin), mixer.v_conv.weight).view_as(_payload)
            payload, write_keys = payload.float(), write_keys.float()
            query_keys = query_keys.float()
            lam_vec, lam_inc = mixer.channel_decay(xin)
            drift = (lam_vec.float() - lam_inc.float().unsqueeze(-1)).abs()
            gate_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(mixer.b_proj(xin)).view(
                    batch, tokens, mixer.heads, mixer.value_dim)
            gw_dist = (1.0 - gate_vec.float()).abs()
            r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(mixer.g_proj(xin)).view(
                    batch, tokens, mixer.heads, mixer.value_dim)
            gr_dist = (r_vec.float() - 1.0).abs()
            state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                mixer.value_dim, dtype=torch.float32,
                                device=xin.device)
            live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               dtype=torch.bool, device=xin.device)
            keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               mixer.key_dim, dtype=torch.float32,
                               device=xin.device)
            out = _k15_forward_launch(
                payload, write_keys, query_keys, lam_vec.float(),
                gate_vec.float(), state, live, keys, pol32, mixer.rank,
                mixer.segment_tokens, True, collect=True, warps=8, stages=3)
        with torch.no_grad():
            stats, _schedule = _k15_walk_stats(
                out, mixer.rank, mixer, payload=payload,
                decay_vec=lam_vec.float())
            per_layer.append((layer_index, stats))
            gate_rows.append(dict(
                layer=layer_index,
                decay_gate=dict(
                    mean_distance_from_registered_scalar=float(drift.mean()),
                    max=float(drift.max())),
                write_gate=dict(
                    mean_distance_from_1=float(gw_dist.float().mean()),
                    max=float(gw_dist.float().max())),
                read_gate=dict(
                    mean_distance_from_1=float(gr_dist.float().mean()),
                    max=float(gr_dist.float().max()))))
            weight = mixer.v_conv.weight.detach()
            identity_birth = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=weight.device).view(1, -1)
            conv_rows.append(dict(
                layer=layer_index,
                tap_means=[float(weight[:, j].float().mean())
                           for j in range(weight.shape[1])],
                tap_stds=[float(weight[:, j].float().std())
                          for j in range(weight.shape[1])],
                kernel_l2=float(weight.float().norm()),
                distance_from_identity_birth=float(
                    (weight - identity_birth).abs().mean()),
                distance_from_identity_birth_max=float(
                    (weight - identity_birth).abs().max())))
    return hook


# ---------------------------------------------- the mechanism instruments


def k24_composed_diagnostics(model, stream, mult: int,
                             windows: int = PROBE_WINDOWS) -> dict:
    """The C5 instrument on the COMPOSED machine at the arm's own window
    length: the K15 walk-stats (effective rows, occupancy, the op mix, the
    identity share, the read-weight entropy) + the p3 gate telemetry, on
    the first ``windows`` FULL windows of the arm's own context (the
    k15_probe_diagnostics hook bodies, collected at the arm's token
    count)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    context = WINDOW_CONTEXT[mult]
    rows = k24_probe_rows(stream, mult, windows)
    model.eval()
    per_layer = []
    gate_rows = []
    handles = [block.mixer.register_forward_hook(
        _composed_hook(per_layer, gate_rows, i, block.mixer))
        for i, block in enumerate(model.blocks)]
    started = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1, context))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    wall = time.monotonic() - started
    summary = _agg_walk(per_layer)
    summary["probe"] = dict(
        split="calibration", windows=len(rows), mixers=len(model.blocks),
        window_mult=mult, context=context,
        regime="eval, bf16 autocast, the k15 composed persistent walk on "
               "the arm's own full windows (collect=True)",
        probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the born-identity "
             "drift for ALL THREE gate channels",
        per_layer={str(r["layer"]): {k: v for k, v in r.items()
                                     if k != "layer"} for r in gate_rows},
        decay_gate_mean_distance_from_registered_scalar=float(sum(
            r["decay_gate"]["mean_distance_from_registered_scalar"]
            for r in gate_rows) / max(1, len(gate_rows))),
        write_gate_mean_distance_from_1=float(sum(
            r["write_gate"]["mean_distance_from_1"] for r in gate_rows)
            / max(1, len(gate_rows))),
        read_gate_mean_distance_from_1=float(sum(
            r["read_gate"]["mean_distance_from_1"] for r in gate_rows)
            / max(1, len(gate_rows))),
        n_layers=len(gate_rows))
    return summary


def k24_hybrid_diagnostics(model, stream, mult: int,
                           windows: int = PROBE_WINDOWS) -> dict:
    """The C5 instrument on the HYBRID's slot layers at the arm's own
    window length (the K22 instrument's hook bodies at the longer token
    count) + the trained read gates' own values."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    context = WINDOW_CONTEXT[mult]
    rows = k24_probe_rows(stream, mult, windows)
    model.eval()
    slot_layers = hybrid_slot_layers(model)
    per_layer = []
    gate_rows = []
    conv_rows = []
    handles = [model.model.blocks[i].layer.gdn2.slot.register_forward_hook(
        _slot_hook(per_layer, gate_rows, conv_rows, i,
                   model.model.blocks[i].layer.gdn2.slot))
        for i in slot_layers]
    started = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1, context))
                model.loss(x, _y)
    finally:
        for handle in handles:
            handle.remove()
    wall = time.monotonic() - started
    summary = _agg_walk(per_layer)
    summary["probe"] = dict(
        split="calibration", windows=len(rows), mixers=len(slot_layers),
        slot_layers=list(slot_layers), window_mult=mult, context=context,
        regime="eval, bf16 autocast, the k15 composed persistent walk on "
               "the hybrid's slot layers' own inputs (collect=True)",
        probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-slot-layer means over the probe windows",
        per_layer={str(r["layer"]): {k: v for k, v in r.items()
                                     if k != "layer"} for r in gate_rows})
    summary["conv_telemetry"] = dict(
        note="the slot mixers' learned conv kernels (per tap, the distance "
             "from the identity birth)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items()
                                     if k != "layer"} for r in conv_rows})
    summary["k22_read_gates"] = {str(i): float(
        model.model.blocks[i].layer.gdn2.read_gate.detach())
        for i in slot_layers}
    return summary


def k24_read_mask_sweep(model, stream, mult: int) -> dict:
    """THE HEADLINE INSTRUMENT at the arm's own window length: each
    layer's store read masked at eval time (the mixer's output IS its
    store read), one layer at a time, plus the all-layers cell (the
    storeless floor) -- the ``side_by_side.k10_read_mask_sweep`` body with
    the K24 own-window eval.  The load-bearing count vs window length is
    the curve this lane exists for.  Ablation effects, not marginals."""
    import torch
    base = evaluate_at(model, stream, mult)["score"]
    n_layers = len(model.blocks)
    per_layer = {}
    for i in range(n_layers):
        handle = model.blocks[i].mixer.register_forward_hook(
            lambda _m, _a, out: torch.zeros_like(out))
        try:
            nll = evaluate_at(model, stream, mult)["score"]
        finally:
            handle.remove()
        per_layer[str(i)] = nll
        print(f"  read-mask layer {i}: nll {nll:.4f} (base {base:.4f})",
              flush=True)
    all_handles = [block.mixer.register_forward_hook(
        lambda _m, _a, out: torch.zeros_like(out))
        for block in model.blocks]
    try:
        storeless = evaluate_at(model, stream, mult)["score"]
    finally:
        for handle in all_handles:
            handle.remove()
    deltas = {k: v - base for k, v in per_layer.items()}
    return dict(
        note=("the decorative sweep at the arm's own window length: one "
              "layer's mixer output (== its store read, projected) zeroed "
              "for the eval pass, one at a time, plus the all-layers cell "
              "(the storeless floor).  The uniformity question's direct "
              "test: does the load-bearing mask light up as windows "
              "grow?  Ablation effects, not marginals."),
        window_mult=mult, context=WINDOW_CONTEXT[mult],
        base_nll=base, masked_nll_per_layer=per_layer,
        all_layers_masked_nll=storeless,
        delta_per_layer=deltas,
        all_layers_delta=storeless - base,
        load_bearing_bar=LOAD_BEARING_BAR,
        load_bearing_layers=sorted(
            (k for k, v in deltas.items() if v >= LOAD_BEARING_BAR),
            key=int),
        load_bearing_count=sum(1 for v in deltas.values()
                               if v >= LOAD_BEARING_BAR))


def k24_slot_read_mask_sweep(model, stream, mult: int) -> dict:
    """THE HEADLINE INSTRUMENT on the hybrid's slot layers at the arm's
    own window length: each slot layer's READ GATE zeroed for the eval
    pass, one at a time, then all three (the trained dense-stack floor) --
    the ``k22_slot_read_mask_sweep`` body with the K24 own-window eval."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    slot_layers = hybrid_slot_layers(model)
    base = evaluate_at(model, stream, mult)["score"]
    per_layer = {}
    for i in slot_layers:
        wrapper = model.model.blocks[i].layer.gdn2
        saved = float(wrapper.read_gate.detach())
        with torch.no_grad():
            wrapper.read_gate.zero_()
        try:
            nll = evaluate_at(model, stream, mult)["score"]
        finally:
            with torch.no_grad():
                wrapper.read_gate.fill_(saved)
        per_layer[str(i)] = nll
        print(f"  slot-read-mask layer {i}: nll {nll:.4f} "
              f"(base {base:.4f}, gate {saved:.6f})", flush=True)
    saved_all = {i: float(
        model.model.blocks[i].layer.gdn2.read_gate.detach())
        for i in slot_layers}
    try:
        with torch.no_grad():
            for i in slot_layers:
                model.model.blocks[i].layer.gdn2.read_gate.zero_()
        all_masked = evaluate_at(model, stream, mult)["score"]
    finally:
        with torch.no_grad():
            for i, value in saved_all.items():
                model.model.blocks[i].layer.gdn2.read_gate.fill_(value)
    return dict(
        note=("the hybrid's decorative sweep at the arm's own window "
              "length: one slot layer's read gate zeroed for the eval pass, "
              "one at a time, plus the all-slots cell.  The horizon "
              "question's direct test: the banked 1x hybrid found 0 of 3 "
              "slot layers >= the 0.02 bar -- do they clear it when the "
              "regime demands state?"),
        window_mult=mult, context=WINDOW_CONTEXT[mult],
        base_nll=base, masked_nll_per_slot_layer=per_layer,
        all_slots_masked_nll=all_masked,
        delta_per_slot_layer={k: v - base for k, v in per_layer.items()},
        all_slots_delta=all_masked - base,
        gates_at_sweep=saved_all,
        load_bearing_bar=LOAD_BEARING_BAR,
        slot_layers_ge_bar=sum(1 for v in per_layer.values()
                               if (v - base) >= LOAD_BEARING_BAR))


def k24_gdn2_telemetry(model, stream, mult: int,
                       windows: int = PROBE_WINDOWS) -> dict:
    """GDN2's own state-saturation telemetry at the arm's own window
    length (the ``side_by_side.gdn2_gate_telemetry`` body, the committed
    arm's own exposed state surface: the retention exp(g), the implied
    memory horizon, the erase/write gates) on the first ``windows`` FULL
    windows of the arm's context."""
    import torch
    import torch.nn.functional as F
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    context = WINDOW_CONTEXT[mult]
    rows = k24_probe_rows(stream, mult, windows)
    cores = [block.layer.gdn2.gdn2 for block in model.model.blocks]
    acc: dict[int, list] = {}

    def make_hook(i):
        def hook(module, args, _output):
            x = args[0]
            with torch.no_grad():
                # the layer's own gate arithmetic (gdn2.py forward, verbatim
                # expressions over the module's own projections)
                g = (-module.A_log.float().exp().repeat_interleave(
                        module.head_k_dim)
                     * F.softplus(module.f_proj(x).float() + module.dt_bias))
                b = module.b_proj(x).sigmoid()
                w = module.w_proj(x).sigmoid()
                retention = g.exp()
                acc.setdefault(i, []).append(dict(
                    mean_retention=float(retention.float().mean()),
                    mean_erase_b=float(b.float().mean()),
                    mean_write_w=float(w.float().mean())))
        return hook

    handles = [core.register_forward_hook(make_hook(i))
               for i, core in enumerate(cores)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                x, y, _v = next(grouped([row], 1, context))
                model.loss(x, y)          # drives the same forward
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    per_layer = {}
    for i in sorted(acc):
        entries = acc[i]
        mean_retention = sum(e["mean_retention"] for e in entries) \
            / len(entries)
        per_layer[str(i)] = dict(
            mean_state_retention_exp_g=mean_retention,
            implied_memory_horizon_tokens=1.0 / max(
                1e-9, 1.0 - mean_retention),
            mean_erase_gate_b=sum(e["mean_erase_b"] for e in entries)
            / len(entries),
            mean_write_gate_w=sum(e["mean_write_w"] for e in entries)
            / len(entries))
    head_decay_base = {}
    for i, core in enumerate(cores):
        head_decay_base[str(i)] = [float(v) for v in
                                   core.A_log.detach().float().exp().cpu()]
    horizons = [row["implied_memory_horizon_tokens"]
                for row in per_layer.values()]
    return dict(
        note=("gdn2's own gate telemetry per layer at the arm's own window "
              "length (the state's write/forget coefficients -- the "
              "committed arm's exposed state surface; the recurrent state "
              "itself is internal to the FLA chunk kernel): the retention "
              "exp(g), its reciprocal-of-complement the implied memory "
              "horizon, the erase and write gates.  The saturation "
              "question's own receipt: does the dense state's horizon "
              "MOVE when the regime demands longer state?"),
        window_mult=mult, context=context, windows=len(rows),
        per_layer=per_layer,
        mean_implied_memory_horizon_tokens=float(
            sum(horizons) / max(1, len(horizons))),
        head_decay_base=head_decay_base)


# ------------------------------------------ the hybrid's at-birth receipts


def k24_hybrid_birth_receipts(model, stream, mult: int, seed: int) -> dict:
    """The K22 birth discipline at the arm's own (long) window, BEFORE any
    training:

    * **born as GDN2, bit for bit**: the hybrid's loss on the arm's own
      first microbatch vs a pure-GDN2 twin of the SAME build on the same
      microbatch -- bit-identical (the read gates born at exact 0.0; the
      only legal escape is the IEEE sign-of-zero corner, invisible to
      the loss).
    * **the open signal**: the read gates' gradients NONZERO at birth.
    * **the inert birth**: every OTHER slot parameter's gradient EXACTLY
      ZERO while a gate is exactly zero.
    * **the store runs at birth**: the slot stores fill (APPEND/CORRECT
      fire) on the same microbatch -- the machinery runs, nothing reads
      out.
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    context = WINDOW_CONTEXT[mult]
    microbatch = MICROBATCH[mult]
    slot_layers = hybrid_slot_layers(model)
    batch_rows = next(iter(stream.batches("train", context=context,
                                          batch=WINDOWS_PER_STEP[mult])))
    x, y, _valid = next(iter(grouped(batch_rows, microbatch, context)))
    twin, _cfg, _summary = matched_arms.build_gdn2(seed)
    with torch.no_grad():
        loss_twin = twin.loss(x, y)
    del twin
    torch.cuda.empty_cache()
    loss = model.loss(x, y)
    loss.backward()
    gates = {}
    inert = {}
    for i in slot_layers:
        wrapper = model.model.blocks[i].layer.gdn2
        gate = wrapper.read_gate
        gates[str(i)] = dict(
            value=float(gate.detach()),
            born_at_exact_zero=bool(gate.detach() == 0.0),
            grad=float(gate.grad.detach()) if gate.grad is not None
            else None)
        slot_grads = {name: param.grad
                      for name, param in wrapper.slot.named_parameters()
                      if param.grad is not None}
        inert[str(i)] = dict(
            tensors_with_grad=len(slot_grads),
            all_exactly_zero=bool(
                all(bool((g == 0).all()) for g in slot_grads.values())),
            max_abs_grad=float(max(
                (g.abs().max().item() for g in slot_grads.values()),
                default=0.0)))
    model.zero_grad(set_to_none=True)
    # the store-run receipt: the ops fire on the slot layers at birth
    per_layer = []
    gate_rows = []
    conv_rows = []
    with torch.no_grad():
        handles = [model.model.blocks[i].layer.gdn2.slot
                   .register_forward_hook(
                       _slot_hook(per_layer, gate_rows, conv_rows, i,
                                  model.model.blocks[i].layer.gdn2.slot))
                   for i in slot_layers]
        try:
            model.loss(x, y)
        finally:
            for handle in handles:
                handle.remove()
    walk = _agg_walk(per_layer)
    ops_fill = bool(all(
        entry["op_rate_per_token"].get("append", 0.0)
        + entry["op_rate_per_token"].get("correct", 0.0) > 0.0
        for _layer, entry in walk["per_layer"].items()))
    return dict(
        probe="k24_hybrid_birth_receipts", seed=seed, window_mult=mult,
        context=context, microbatch=microbatch,
        born_as_gdn2=dict(
            hybrid_loss=float(loss.detach()),
            pure_twin_loss=float(loss_twin),
            bit_identical=bool(float(loss.detach()) == float(loss_twin)),
            note="the hybrid vs a pure-GDN2 twin of the same build on the "
                 "arm's own first long-window microbatch: the read gates "
                 "are born at exact 0.0, so the machine IS the committed "
                 "GDN2 arm at birth, at the LONG window"),
        read_gates=gates,
        open_signal=dict(
            note="dLoss/dgate x slot content at birth: the gate's own "
                 "gradient (the open signal) is nonzero at birth while "
                 "every other slot parameter receives an exactly-zero "
                 "gradient (inert at birth)",
            all_gates_born_at_exact_zero=all(
                g["born_at_exact_zero"] for g in gates.values()),
            any_gate_gradient_nonzero=any(
                (g["grad"] or 0.0) != 0.0 for g in gates.values())),
        inert_at_birth=dict(
            per_layer=inert,
            all_inert=bool(all(v["all_exactly_zero"]
                               for v in inert.values()))),
        store_runs_at_birth=dict(
            note="the slot stores fill at birth (the machinery runs; "
                 "nothing reads out until a gate opens)",
            ops_fill=ops_fill,
            per_layer_op_rate={str(layer): entry["op_rate_per_token"]
                               for layer, entry
                               in walk.get("per_layer", {}).items()},
            occupancy={str(layer): entry["occupancy_mean_live_slots"]
                       for layer, entry
                       in walk.get("per_layer", {}).items()}))


# --------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import SLOT_LAYERS

    arm = ARMS[args.arm]
    machine = arm["machine"]
    mult = arm["mult"]
    seed = args.seed
    context = WINDOW_CONTEXT[mult]
    windows_per_step = WINDOWS_PER_STEP[mult]
    microbatch = MICROBATCH[mult]
    streams, identity = make_streams()
    stream = streams[mult]
    card = _require_lease()

    # ---- the window arithmetic receipt (computed from the manifest,
    # asserted against the registration's committed receipt)
    registration_receipt_path = RECEIPTS / "k24_window_arithmetic.json"
    window_receipt = window_arithmetic(Path(identity["manifest"]), mult)
    if registration_receipt_path.exists() and not args.smoke:
        registered = json.loads(registration_receipt_path.read_text())
        if registered["arms"][str(mult)] != window_receipt:
            raise ValueError(
                f"the window arithmetic drifted from the registration's "
                f"committed receipt at mult {mult} -- the manifest changed "
                f"under the lane (escalate, not excuse)")

    # ---- the machine: the committed build VERBATIM
    model, optimizer, optimizer_config, routing_receipt, optimizer_block, \
        summary = build_machine(machine, seed)
    parameters_total = sum(p.numel() for p in model.parameters())
    expected_parameters = {"ours": 158_638_848, "gdn2": 149_855_670,
                          "hybrid": 152_517_585}[machine]
    if parameters_total != expected_parameters:
        raise ValueError(f"the {machine} machine's parameters must be the "
                         f"committed {expected_parameters}; got "
                         f"{parameters_total}")

    # ---- the anchors' banked rows (loaded from the committed receipts)
    anchor = None
    if arm["role"] == "anchor":
        if mult != 1:
            raise ValueError("an anchor arm must be a 1x cell")
        anchor = ours_1x_banked() if machine == "ours" else gdn2_1x_banked()
    banked_hybrid = hybrid_1x_banked() if machine == "hybrid" else None

    # ---- the checkpoint grid: the banked grid PLUS the predicted
    # constant-phase landing step (the EMA/diagnostic states are read at
    # the landing; the 2x/4x landings are 182/178, off the banked grid)
    checkpoint_steps = tuple(sorted(
        set(args.checkpoint_steps) | {window_receipt["landing_step"]}))

    run_config = dict(
        probe="k24_long_window", arm=args.arm, machine=machine,
        window_mult=mult, context=context,
        windows_per_step=windows_per_step, microbatch=microbatch,
        tokens_per_microbatch=microbatch * context,
        microbatch_groups_per_step=-(-windows_per_step // microbatch),
        seed=seed, arm_note=arm["note"], role=arm["role"],
        corrected_scoring=CORRECTED_SCORING, rules=RULES, bands=BANDS,
        exposure_targets=int(args.bytes),
        step_shape=dict(
            note="the probe's step shape, adjusted per the registration so "
                 "steps x tokens consume the same 20M targets: the "
                 "~111K-targets/step shape and the ~180-step landing are "
                 "HELD (windows/step scaled down as the context grows), "
                 "and the microbatch's TOKEN count is held at the banked "
                 "16,384 (the memory-matched rule); matched EXPOSURE is "
                 "the invariant",
            window_arithmetic=window_receipt),
        window_discipline=RULES["boundaries"]["text"],
        eval_rule=("the arm's own-window eval: the first 7 FULL windows "
                   "of the arm's context in calibration order (at 1x this "
                   "IS the banked L.evaluate set, verbatim); plus the "
                   "cross-eval grid at all three window lengths on the "
                   "post-cooldown final state"),
        machine_build=dict(
            ours="velocity_ladder._build_k10 verbatim + "
                 "k15_composition.convert_model_to_composition (the banked "
                 "1.3811 machine's own build)",
            gdn2="matched_arms.build_gdn2 verbatim (the banked 1.3522 "
                 "machine's own build)",
            hybrid="matched_arms.build_gdn2 verbatim + "
                   "k22_hybrid.convert_gdn2_to_hybrid (the K22 engine "
                   "VERBATIM: the composed slot mixers at {6, 12, 18} "
                   "through the zero-born read gates)")[machine],
        optimizer=optimizer_block,
        routing="the machine's own banked routing split, imported verbatim "
                "(ours: k11b_probe_channel_gates.split_parameters_gated "
                "'p3'; gdn2: muon_control.split_parameters_gdn2 option B; "
                "hybrid: k22_hybrid.split_parameters_k22)",
        routing_receipt=routing_receipt,
        slot_layers=(list(SLOT_LAYERS) if machine == "hybrid" else None),
        parameters=parameters_total,
        checkpoint_steps=list(checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe stopping rule, "
                       f"constant LR), THEN the {COOLDOWN_STEPS}-step "
                       "linear cooldown (the corrected-scoring extension, "
                       "same stream); the honest elapsed wall is reported "
                       "first-class"),
        anchors=dict(
            ours_1x=(None if not (machine == "ours"
                                  and arm["role"] == "anchor") else
                     {k: anchor[k] for k in ("post_cooldown", "constant_lr",
                                            "ema180", "step1",
                                            "exposure_bytes", "total_bytes",
                                            "steps", "run_dir")}),
            gdn2_1x=(None if not (machine == "gdn2"
                                  and arm["role"] == "anchor") else
                     {k: anchor[k] for k in ("post_cooldown", "constant_lr",
                                            "ema180", "step1",
                                            "exposure_bytes", "total_bytes",
                                            "steps", "run_dir")}),
            hybrid_1x=banked_hybrid,
            composed_ours_banked=composed_4x_banked(),
            note="the 1x cells MUST reproduce the banked rows bit for bit "
                 "FIRST; any mismatch STOPS the lane and escalates"),
        bindings=k24_bindings(), summary=summary, identity=identity,
        versions=_versions(), git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=parameters_total,
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm=args.arm, machine=machine, mult=mult,
                          card=card, seed=seed,
                          parameters=parameters_total,
                          context=context,
                          windows_per_step=windows_per_step,
                          microbatch=microbatch,
                          peak_lr=optimizer_block["peak_lr"])), flush=True)

    # ---- the hybrid's at-birth receipts at the long window, BEFORE any
    # training (the K22 discipline)
    birth = None
    if machine == "hybrid":
        birth = k24_hybrid_birth_receipts(model, stream, mult, seed)
        run_config["birth_receipts"] = birth
        if not args.smoke and not (
                birth["born_as_gdn2"]["bit_identical"]
                and birth["open_signal"]["all_gates_born_at_exact_zero"]
                and birth["open_signal"]["any_gate_gradient_nonzero"]
                and birth["inert_at_birth"]["all_inert"]
                and birth["store_runs_at_birth"]["ops_fill"]):
            raise ValueError("the BORN-AS-GDN2 receipts failed at the "
                             "long window -- escalate, not excuse (the "
                             "birth discipline is non-negotiable)")
        print("  born-as-GDN2 at the long window: loss bit-identical "
              f"{birth['born_as_gdn2']['bit_identical']}; open signal "
              f"{birth['open_signal']['any_gate_gradient_nonzero']}; "
              f"slot inert {birth['inert_at_birth']['all_inert']}; "
              f"store fills "
              f"{birth['store_runs_at_birth']['ops_fill']}", flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    banked_rows = None if args.smoke or anchor is None else anchor["rows"]
    run = _train_corrected_k24(
        model, optimizer, optimizer_block["peak_lr"],
        optimizer_config["warmup_steps"], microbatch, stream, context,
        windows_per_step, int(args.bytes), f"{args.arm}_seed{seed}",
        out / "training.jsonl", checkpoint_steps, cooldown_steps, EMA_DECAY,
        banked_rows=banked_rows)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the landing must be the receipt's own predicted landing (the
    # window arithmetic is deterministic from the manifest; fail closed)
    if not args.smoke and run["constant_steps"] \
            != window_receipt["landing_step"]:
        raise ValueError(
            f"the constant-phase landing {run['constant_steps']} != the "
            f"window receipt's predicted "
            f"{window_receipt['landing_step']} -- the stream arithmetic "
            f"drifted (escalate, not excuse)")

    # ---- the anchor's landing bar: the banked row reproduced bit for bit
    anchor_receipt = None
    if anchor is not None and not args.smoke:
        model.load_state_dict(run["snapshots"]["final"])
        os.environ["X40_BATCHED_READS"] = "0"
        landing = L.evaluate(model, stream, "matrix_ops", "calibration")
        checks = dict(
            steps=len(run["rows"]),
            steps_match=bool(len(run["rows"]) == len(anchor["rows"])),
            trajectory_bit_identical=bool(all(
                r["loss"] == b["loss"] for r, b in
                zip(run["rows"], anchor["rows"]))),
            exposure_match=bool(
                rows_by_step[run["constant_steps"]]["bytes"]
                == anchor["exposure_bytes"]),
            total_bytes_match=bool(run["seen"] == anchor["total_bytes"]),
            landing_bit_identical=bool(
                landing["score"] == anchor["post_cooldown"]),
            expected_post_cooldown=anchor["post_cooldown"])
        anchor_receipt = dict(
            note="THE ANCHOR CONVENTION: this 1x cell must reproduce the "
                 "banked row bit for bit (every training-log row, the "
                 "exposure landing, the post-cooldown eval); any mismatch "
                 "STOPS the lane and escalates",
            run_dir=anchor["run_dir"], checks=checks,
            all_checks=bool(all(v for v in checks.values()
                                if isinstance(v, bool))))
        print(json.dumps(dict(anchor_checks=checks)), flush=True)
        if not anchor_receipt["all_checks"]:
            raise ValueError(
                f"ANCHOR MISMATCH on {args.arm}: {checks} -- the 1x cell "
                f"does NOT reproduce the banked row bit for bit (STOP and "
                f"escalate; the anchor convention is non-negotiable)")

    # ---- the scoring pass (restored snapshots; training is over)
    scores = []
    for step in checkpoint_steps:                  # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        result = evaluate_at(model, stream, mult)
        row = dict(kind="constant", step=step,
                   bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f}",
              flush=True)
    if run["snapshots"] and run["ema"] is not None and \
            run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(run["ema"][name].to(param.dtype))
        result = evaluate_at(model, stream, mult)
        scores.append(dict(kind="ema_constant", step=run["constant_steps"],
                           bytes=rows_by_step[run["constant_steps"]]["bytes"],
                           eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"],
                           ema_decay=EMA_DECAY,
                           note="ranking-only: biased toward mid-run weights"))
        print(f"  EMA at step {run['constant_steps']}: "
              f"eval {result['score']:.4f}", flush=True)
    post_cooldown = None
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        result = evaluate_at(model, stream, mult)
        post_cooldown = result["score"]
        scores.append(dict(kind="post_cooldown", step=run["step"],
                           bytes=run["seen"], eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"]))
        print(f"  post-cooldown final (step {run['step']}): "
              f"eval {result['score']:.4f}", flush=True)
    with (out / "scores.jsonl").open("w", buffering=1) as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    # ---- the anchor's constant/EMA bar (the banked row's own numbers,
    # every digit -- asserted after the scoring pass computed them)
    if anchor is not None and not args.smoke:
        constant_final = next(
            (r["eval_nll"] for r in scores if r["kind"] == "constant"
             and r["step"] == run["constant_steps"]), None)
        ema_final = next((r["eval_nll"] for r in scores
                          if r["kind"] == "ema_constant"), None)
        if constant_final != anchor["constant_lr"] \
                or ema_final != anchor["ema180"]:
            raise ValueError(
                f"ANCHOR MISMATCH on {args.arm}: constant-180 "
                f"{constant_final!r} (banked {anchor['constant_lr']!r}) / "
                f"EMA-180 {ema_final!r} (banked {anchor['ema180']!r}) -- "
                f"the 1x cell does NOT reproduce the banked row's scores "
                f"bit for bit (STOP and escalate)")
        anchor_receipt["checks"]["constant_bit_identical"] = \
            bool(constant_final == anchor["constant_lr"])
        anchor_receipt["checks"]["ema_bit_identical"] = \
            bool(ema_final == anchor["ema180"])
        anchor_receipt["all_checks"] = bool(all(
            v for v in anchor_receipt["checks"].values()
            if isinstance(v, bool)))
        print(json.dumps(dict(anchor_scores=dict(
            constant=constant_final, ema=ema_final,
            banked_constant=anchor["constant_lr"],
            banked_ema=anchor["ema180"])), indent=1), flush=True)

    # ---- the cross-eval grid: the post-cooldown final state evaluated at
    # ALL THREE window lengths (the conditioning-vs-regime decomposition)
    cross_grid = {}
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        for other_mult in WINDOW_MULTS:
            row = evaluate_at(model, streams[other_mult], other_mult)
            cross_grid[str(other_mult)] = dict(
                eval_nll=row["score"], eval_windows=row["eval_windows"],
                eval_targets=row["eval_targets"])
            print(f"  cross-eval at {other_mult}x "
                  f"({WINDOW_CONTEXT[other_mult]}B): "
                  f"{row['score']:.4f}", flush=True)

    # ---- the mechanism receipts at the constant-landing and final states
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    telemetry_180 = telemetry_final = None
    gates_180 = gates_final = None
    if run["status"] == "completed":
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            if machine == "ours":
                diagnostics_180 = k24_composed_diagnostics(
                    model, stream, mult, windows=args.probe_windows)
                sweep_180 = k24_read_mask_sweep(model, stream, mult)
            elif machine == "hybrid":
                diagnostics_180 = k24_hybrid_diagnostics(
                    model, stream, mult, windows=args.probe_windows)
                sweep_180 = k24_slot_read_mask_sweep(model, stream, mult)
                gates_180 = {str(i): float(
                    model.model.blocks[i].layer.gdn2.read_gate.detach())
                    for i in SLOT_LAYERS}
            else:
                telemetry_180 = k24_gdn2_telemetry(
                    model, stream, mult, windows=args.probe_windows)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            if machine == "ours":
                diagnostics_final = k24_composed_diagnostics(
                    model, stream, mult, windows=args.probe_windows)
                sweep_final = k24_read_mask_sweep(model, stream, mult)
            elif machine == "hybrid":
                diagnostics_final = k24_hybrid_diagnostics(
                    model, stream, mult, windows=args.probe_windows)
                sweep_final = k24_slot_read_mask_sweep(model, stream, mult)
                gates_final = {str(i): float(
                    model.model.blocks[i].layer.gdn2.read_gate.detach())
                    for i in SLOT_LAYERS}
            else:
                telemetry_final = k24_gdn2_telemetry(
                    model, stream, mult, windows=args.probe_windows)

    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    ema_final = next((r["eval_nll"] for r in scores
                      if r["kind"] == "ema_constant"), None)
    receipt = dict(
        probe="k24_long_window", arm=args.arm, machine=machine,
        window_mult=mult, context=context,
        windows_per_step=windows_per_step, microbatch=microbatch,
        seed=seed, arm_note=arm["note"], role=arm["role"],
        corrected_scoring=CORRECTED_SCORING, status=run["status"],
        window_arithmetic=window_receipt,
        exposure_bytes=rows_by_step[run["constant_steps"]]["bytes"],
        exposure_steps=run["constant_steps"],
        total_bytes=run["seen"], total_steps=run["step"],
        cooldown_steps=cooldown_steps,
        train_seconds_constant=round(run.get("constant_elapsed", 0.0), 1),
        train_seconds_total=round(run["elapsed"], 1),
        bytes_per_s=(rows_by_step[run["constant_steps"]]["bytes"]
                     / max(1e-9, run.get("constant_elapsed", 1.0))),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        parameters=parameters_total,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cross_eval_grid=cross_grid,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        anchor_receipt=anchor_receipt,
        birth_receipts=birth,
        gate_trajectory=dict(
            init=({str(i): 0.0 for i in SLOT_LAYERS}
                  if machine == "hybrid" else None),
            constant=gates_180, final=gates_final),
        scores=scores,
        anchors=run_config["anchors"],
        diagnostics_step180=diagnostics_180 if machine != "gdn2" else None,
        diagnostics_post_cooldown=(diagnostics_final
                                   if machine != "gdn2" else None),
        read_mask_sweep_step180=(sweep_180 if machine == "ours" else None),
        read_mask_sweep_post_cooldown=(sweep_final
                                       if machine == "ours" else None),
        slot_read_mask_sweep_step180=(sweep_180 if machine == "hybrid"
                                      else None),
        slot_read_mask_sweep_post_cooldown=(sweep_final
                                            if machine == "hybrid" else None),
        gdn2_telemetry_step180=telemetry_180,
        gdn2_telemetry_post_cooldown=telemetry_final,
        routing_receipt=routing_receipt,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            run["rows"], keep_first=min(5, max(1, len(run["rows"])))))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "machine", "window_mult", "status",
                       "exposure_steps", "exposure_bytes",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt


# ------------------------------------------------------------------- assemble


def _arm_row(path: Path) -> dict | None:
    if not (path / "result.json").exists():
        return None
    r = json.loads((path / "result.json").read_text())
    sweep = (r.get("read_mask_sweep_post_cooldown")
             or r.get("slot_read_mask_sweep_post_cooldown") or {})
    diag = r.get("diagnostics_post_cooldown") or {}
    tele = r.get("gdn2_telemetry_post_cooldown") or {}
    return dict(
        arm=r["arm"], machine=r["machine"], window_mult=r["window_mult"],
        seed=r["seed"], status=r["status"],
        context=r["context"], windows_per_step=r["windows_per_step"],
        exposure_bytes=r["exposure_bytes"],
        exposure_steps=r["exposure_steps"],
        train_seconds_constant=r["train_seconds_constant"],
        train_seconds_total=r["train_seconds_total"],
        post_cooldown=r["final_eval_nll_post_cooldown"],
        constant_lr=r["final_eval_nll_constant_lr"],
        ema=r["ema_eval_nll"], cross_eval_grid=r.get("cross_eval_grid"),
        window_arithmetic=r.get("window_arithmetic"),
        anchor_checks=(r.get("anchor_receipt") or {}).get("checks"),
        sweep=dict(
            base_nll=sweep.get("base_nll"),
            delta_per_layer=sweep.get("delta_per_layer")
            or sweep.get("delta_per_slot_layer"),
            all_masked_delta=sweep.get("all_layers_delta")
            or sweep.get("all_slots_delta"),
            load_bearing_count=sweep.get("load_bearing_count"),
            load_bearing_layers=sweep.get("load_bearing_layers"),
            slot_layers_ge_bar=sweep.get("slot_layers_ge_bar"),
            gates_at_sweep=sweep.get("gates_at_sweep")),
        mechanism=dict(
            effective_rows=diag.get("effective_read_rows"),
            occupancy=(diag.get("occupancy") or {}).get("mean_live_slots"),
            identity_share=diag.get("identity_rewrite_share"),
            op_mix=diag.get("op_rate_per_token_lane0"),
            per_layer_op_mix={str(k): v["op_rate_per_token"]
                              for k, v in (diag.get("per_layer")
                                           or {}).items()},
            gate_telemetry=diag.get("gate_telemetry"),
            read_gates=diag.get("k22_read_gates")),
        gdn2_telemetry=dict(
            mean_implied_horizon=tele.get(
                "mean_implied_memory_horizon_tokens"),
            per_layer={k: dict(
                retention=v["mean_state_retention_exp_g"],
                horizon=v["implied_memory_horizon_tokens"],
                erase_b=v["mean_erase_gate_b"],
                write_w=v["mean_write_gate_w"])
                for k, v in (tele.get("per_layer") or {}).items()}),
        gate_trajectory=r.get("gate_trajectory"),
        birth_receipts=r.get("birth_receipts"))


def assemble(args) -> dict:
    """The curves and the pre-stated clause adjudications: the
    gap-vs-window curve, the load-bearing-count-vs-window curve (THE
    headline), the effective-window telemetry, the mechanism rows, and the
    verdict on the horizon hypothesis at probe scale -- every delta
    stated against the 0.0119 winner-seed band (no band-worship)."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    rows = {}
    for label in ARMS:
        row = _arm_row(Path(args.runs) / f"{label}_seed{PRIMARY_SEED}")
        if row is not None:
            rows[label] = row
    banked_ours = ours_1x_banked()
    banked_gdn2 = gdn2_1x_banked()
    banked_hybrid = hybrid_1x_banked()

    def own(mult, machine):
        return rows.get(f"{machine}_{mult}x")

    # ---- the gap-vs-window curve (L-2's object)
    gap_curve = {}
    for mult in WINDOW_MULTS:
        o, g = own(mult, "ours"), own(mult, "gdn2")
        if o is None or g is None or o["post_cooldown"] is None \
                or g["post_cooldown"] is None:
            continue
        gap = o["post_cooldown"] - g["post_cooldown"]
        gap_curve[f"{mult}x"] = dict(
            ours=o["post_cooldown"], gdn2=g["post_cooldown"], gap=gap,
            gap_delta_vs_1x=None, inside_seed_band=None,
            eval_note=f"own-window eval at {mult}x "
                      f"({WINDOW_CONTEXT[mult]}-byte full windows)")
    if "1x" in gap_curve:
        base = gap_curve["1x"]["gap"]
        for key, row in gap_curve.items():
            row["gap_delta_vs_1x"] = row["gap"] - base
            row["inside_seed_band"] = bool(
                abs(row["gap_delta_vs_1x"]) < WINNER_SEED_DELTA)
    # the fixed-1x-eval decomposition: every arm's post-cooldown final
    # state evaluated at the 1x windows (the conditioning removed)
    fixed_eval_gap = {}
    for mult in WINDOW_MULTS:
        o, g = own(mult, "ours"), own(mult, "gdn2")
        if o is None or g is None:
            continue
        o1 = (o["cross_eval_grid"] or {}).get("1", {}).get("eval_nll")
        g1 = (g["cross_eval_grid"] or {}).get("1", {}).get("eval_nll")
        if o1 is not None and g1 is not None:
            fixed_eval_gap[f"{mult}x"] = dict(
                ours_at_1x_eval=o1, gdn2_at_1x_eval=g1,
                gap_at_1x_eval=o1 - g1)

    # ---- the load-bearing-count-vs-window curve (L-1's object, THE
    # headline)
    load_bearing_curve = {}
    for mult in WINDOW_MULTS:
        o = own(mult, "ours")
        if o is not None and o["sweep"].get("load_bearing_count") is not None:
            load_bearing_curve[f"ours_{mult}x"] = dict(
                count=o["sweep"]["load_bearing_count"],
                layers=o["sweep"]["load_bearing_layers"],
                bar=LOAD_BEARING_BAR,
                delta_per_layer=o["sweep"]["delta_per_layer"],
                all_layers_delta=o["sweep"]["all_masked_delta"])
    h4 = own(4, "hybrid")
    hybrid_cells = None
    if h4 is not None:
        hybrid_cells = dict(
            at_4x=dict(
                post_cooldown=h4["post_cooldown"],
                slot_layers_ge_bar=h4["sweep"].get("slot_layers_ge_bar"),
                delta_per_slot_layer=h4["sweep"].get("delta_per_layer"),
                all_slots_delta=h4["sweep"].get("all_masked_delta"),
                gates_final=(h4["gate_trajectory"] or {}).get("final"),
                gates_at_sweep=h4["sweep"].get("gates_at_sweep"),
                mechanism=dict(
                    effective_rows=h4["mechanism"]["effective_rows"],
                    occupancy=h4["mechanism"]["occupancy"],
                    identity_share=h4["mechanism"]["identity_share"],
                    op_mix=h4["mechanism"]["op_mix"])),
            at_1x_banked=dict(
                post_cooldown=banked_hybrid["post_cooldown"],
                slot_layers_ge_bar=banked_hybrid[
                    "slot_read_mask_sweep"]["slot_layers_ge_bar"],
                delta_per_slot_layer=banked_hybrid[
                    "slot_read_mask_sweep"]["delta_per_slot_layer"],
                all_slots_delta=banked_hybrid[
                    "slot_read_mask_sweep"]["all_slots_delta"],
                gates_final=banked_hybrid["gates_final"],
                mechanism=banked_hybrid["mechanism"]))

    # ---- the effective-window telemetry per arm (the honest lever)
    window_telemetry = {
        f"{row['machine']}_{row['window_mult']}x":
            dict(context=row["context"],
                 effective=row["window_arithmetic"][
                     "effective_window_bytes"],
                 landing_step=row["exposure_steps"],
                 exposure=row["exposure_bytes"])
        for row in rows.values()}

    # ---- the mechanism-vs-window rows
    mechanism_curve = {}
    for mult in WINDOW_MULTS:
        o = own(mult, "ours")
        if o is not None:
            mechanism_curve[f"ours_{mult}x"] = dict(
                effective_rows=o["mechanism"]["effective_rows"],
                occupancy=o["mechanism"]["occupancy"],
                identity_share=o["mechanism"]["identity_share"],
                op_mix=o["mechanism"]["op_mix"],
                decay_gate=(o["mechanism"]["gate_telemetry"] or {}).get(
                    "decay_gate_mean_distance_from_registered_scalar"),
                write_gate=(o["mechanism"]["gate_telemetry"] or {}).get(
                    "write_gate_mean_distance_from_1"),
                read_gate=(o["mechanism"]["gate_telemetry"] or {}).get(
                    "read_gate_mean_distance_from_1"))
        g = own(mult, "gdn2")
        if g is not None:
            horizons = [v["horizon"] for v in
                        g["gdn2_telemetry"]["per_layer"].values()]
            retentions = [v["retention"] for v in
                          g["gdn2_telemetry"]["per_layer"].values()]
            mechanism_curve[f"gdn2_{mult}x"] = dict(
                mean_implied_horizon_tokens=(float(sum(horizons)
                                                   / max(1, len(horizons)))
                                             if horizons else None),
                mean_retention=(float(sum(retentions)
                                      / max(1, len(retentions)))
                                if retentions else None),
                per_layer_horizons=horizons)

    # ---- the pre-stated clause adjudications
    l1 = dict(rule=RULES["L1"],
              banked_1x_count=load_bearing_curve.get("ours_1x", {}).get(
                  "count"),
              counts={k: v["count"] for k, v in
                      load_bearing_curve.items()})
    if load_bearing_curve.get("ours_4x") and \
            load_bearing_curve.get("ours_1x"):
        rose = load_bearing_curve["ours_4x"]["count"] \
            - load_bearing_curve["ours_1x"]["count"]
        l1["count_rise_4x_vs_1x"] = rose
        l1["mask_lit_up"] = bool(rose > 0)
    if hybrid_cells is not None:
        l1["hybrid_slot_layers_ge_bar"] = dict(
            at_1x=hybrid_cells["at_1x_banked"]["slot_layers_ge_bar"],
            at_4x=hybrid_cells["at_4x"]["slot_layers_ge_bar"])
        l1["hybrid_mask_lit_up"] = bool(
            hybrid_cells["at_4x"]["slot_layers_ge_bar"]
            > hybrid_cells["at_1x_banked"]["slot_layers_ge_bar"])
        l1["hybrid_slot_occupancy_vs_1x_banked"] = dict(
            at_1x_banked=hybrid_cells["at_1x_banked"]["mechanism"][
                "occupancy"],
            at_4x=hybrid_cells["at_4x"]["mechanism"]["occupancy"])
        l1["hybrid_slot_effective_rows_vs_1x_banked"] = dict(
            at_1x_banked=hybrid_cells["at_1x_banked"]["mechanism"][
                "effective_rows"],
            at_4x=hybrid_cells["at_4x"]["mechanism"]["effective_rows"])
        ours_rows_rise = None
        if all(own(m, "ours") for m in WINDOW_MULTS):
            ours_rows_rise = {
                f"{m}x": own(m, "ours")["mechanism"]["effective_rows"]
                for m in WINDOW_MULTS}
        l1["ours_effective_rows_by_window"] = ours_rows_rise
        l1["ours_occupancy_by_window"] = {
            f"{m}x": own(m, "ours")["mechanism"]["occupancy"]
            for m in WINDOW_MULTS if own(m, "ours")}
    l1["held"] = bool(l1.get("mask_lit_up")
                      or (hybrid_cells is not None
                          and l1.get("hybrid_mask_lit_up")))
    l2 = dict(rule=RULES["L2"], banked_gap_1x=None, gap_curve=gap_curve,
              fixed_1x_eval_gap=fixed_eval_gap)
    if "1x" in gap_curve:
        l2["banked_gap_1x"] = gap_curve["1x"]["gap"]
    if "4x" in gap_curve and "1x" in gap_curve:
        delta = gap_curve["4x"]["gap"] - gap_curve["1x"]["gap"]
        l2["gap_change_4x_vs_1x"] = delta
        l2["inside_seed_band"] = bool(abs(delta) < WINNER_SEED_DELTA)
        l2["shrinks"] = bool(delta < 0)
        l2["held"] = bool(delta < -WINNER_SEED_DELTA)
    refutation = None
    if "4x" in gap_curve and "ours_4x" in load_bearing_curve \
            and "ours_1x" in load_bearing_curve:
        mask_dark = bool(load_bearing_curve["ours_4x"]["count"]
                         <= load_bearing_curve["ours_1x"]["count"])
        gap_flat = l2.get("inside_seed_band", True)
        if mask_dark and gap_flat:
            refutation = ("REFUTED AT PROBE SCALE: the mask stayed dark "
                          f"({load_bearing_curve['ours_4x']['count']} "
                          f"layers >= the bar at 4x vs "
                          f"{load_bearing_curve['ours_1x']['count']} at "
                          "1x) and the gap stayed flat at 4x (inside the "
                          "0.0119 seed band) -- the horizon hypothesis is "
                          "dead at probe scale; the board run's horizon "
                          "rationale goes with it, and the remaining board "
                          "value is scale/exposure alone")
        elif l1.get("held") or l2.get("held"):
            refutation = ("NOT REFUTED at probe scale: the pre-stated "
                          "directional clauses moved (receipted per "
                          "clause, every delta against the 0.0119 band)")
        else:
            refutation = ("MIXED: the mask and the gap moved in opposite "
                          "directions or inside the band -- receipted "
                          "per clause; no clause clears its bar and the "
                          "refutation clause does not fire cleanly "
                          "(the conservative reading: no horizon claim "
                          "banked at probe scale)")

    # ---- the escalation flags (the operator hears it from the receipts)
    escalation_flags = []
    for label, row in rows.items():
        if row["anchor_checks"] is not None \
                and row["status"] == "completed" \
                and not row["anchor_checks"].get("landing_bit_identical"):
            escalation_flags.append(
                f"ANCHOR MISMATCH on {label} -- the lane should have "
                f"stopped (this flag is the receipt's own record)")
    if "4x" in gap_curve and gap_curve["4x"]["gap"] < 0:
        escalation_flags.append(
            "4x INVERSION: ours BEATS gdn2 at 4x on the own-window "
            "post-cooldown rows -- the banked conclusion (GDN2 ahead at "
            "probe scale) INVERTS at the long window; the operator hears "
            "it from the receipts")
    if hybrid_cells is not None \
            and hybrid_cells["at_4x"]["slot_layers_ge_bar"] > 0:
        escalation_flags.append(
            "4x INVERSION: the hybrid's slot layers became LOAD-BEARING at "
            "4x (the banked 1x uniformity finding does not extend to the "
            "long-window regime); the operator hears it from the receipts")

    verdict = dict(
        probe="k24_long_window assemble",
        corrected_scoring=CORRECTED_SCORING,
        rules=RULES, bands=BANDS,
        banked=dict(
            ours_1x={k: banked_ours[k] for k in
                     ("post_cooldown", "constant_lr", "ema180", "step1",
                      "exposure_bytes", "run_dir")},
            gdn2_1x={k: banked_gdn2[k] for k in
                     ("post_cooldown", "constant_lr", "ema180", "step1",
                      "exposure_bytes", "run_dir")},
            hybrid_1x=banked_hybrid),
        arms=rows,
        window_telemetry=window_telemetry,
        gap_vs_window=gap_curve,
        fixed_1x_eval_gap=fixed_eval_gap,
        load_bearing_vs_window=load_bearing_curve,
        hybrid_slot_cells=hybrid_cells,
        mechanism_vs_window=mechanism_curve,
        L1=l1, L2=l2, refutation=refutation,
        escalation_flags=escalation_flags)
    path = Path(args.runs) / "assemble.json"
    atomic_json(path, verdict)
    print(json.dumps(dict(
        arms={k: dict(post_cooldown=v["post_cooldown"],
                      constant=v["constant_lr"])
              for k, v in rows.items()},
        gap_vs_window={k: v["gap"] for k, v in gap_curve.items()},
        load_bearing={k: v["count"] for k, v in
                      load_bearing_curve.items()},
        L1_held=l1.get("held"), L2_held=l2.get("held"),
        refutation=refutation,
        escalation_flags=escalation_flags), indent=1, default=str),
        flush=True)
    return verdict


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True, choices=sorted(ARMS))
    a.add_argument("--seed", type=int, default=PRIMARY_SEED)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--output", required=True)
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    a.add_argument("--smoke", action="store_true",
                   help="the pipeline shakedown: tiny exposure, 3-step "
                        "cooldown, no anchor or birth assertions")
    w = sub.add_parser("window_receipt")
    w.add_argument("--out", default=str(RECEIPTS
                                        / "k24_window_arithmetic.json"))
    s = sub.add_parser("assemble")
    s.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args()
    if args.cmd == "window_receipt":
        from experiments.expressivity_tasks.x42_pile_scale.velocity_state \
            import atomic_json
        receipt = registration_window_receipt()
        atomic_json(Path(args.out), receipt)
        print(json.dumps(receipt["arms"], indent=1), flush=True)
        return
    if args.cmd == "arm":
        if args.seed != PRIMARY_SEED:
            raise ValueError(f"the bracket runs the standing seed "
                             f"{PRIMARY_SEED} (got {args.seed})")
        if args.smoke:
            args.bytes = 300_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        run_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
