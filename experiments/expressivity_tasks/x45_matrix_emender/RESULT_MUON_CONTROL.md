# RESULT_MUON_CONTROL.md — the symmetric control on the Muon lead claim

**Date:** 2026-11-20 · **Status: complete — the lead is RETRACTED.** ·
**Authorisation:** the operator's symmetric-control directive — the bake-off
(`RESULT_OPTIMIZER_BAKEOFF.md`) measured hybrid Muon at 3× LR on OUR winner
at **1.5146** vs GDN2's banked AdamW row **1.5837**, a 0.069 lead at matched
exposure, but GDN2 never got a Muon arm: the lead was our-best-optimizer vs
GDN2's-AdamW. This control closes that hole. The GDN2 routing convention
(OPTION A, Muon on every 2D hidden-layer matrix) was approved by the
supervisor before launch; the verdict rule was pre-stated with that approval.

## 0. The verdict, stated first

**RETRACTED. GDN2 + hybrid Muon at 3× its own AdamW LR lands at 1.4702 —
BELOW our winner's Muon row (1.5146) by 0.0444. The bake-off's lead claim
("our winner BEATS gdn2 by 0.069") does not survive the symmetric control.
The honest result is: Muon helps both machines; GDN2 remains ahead (or at
parity within the endpoint-variance band under the maximal-Muon routing) —
the best measured number on this box at this protocol is now GDN2's, not
ours.**

| arm (all at the exact probe protocol, 20,022,866 targets) | final NLL | vs GDN2's AdamW 1.5837 |
|---|---:|---:|
| GDN2, AdamW (banked row 5) | 1.5837 | — |
| our winner, AdamW (banked incumbent) | 1.8227 | +0.2391 |
| our winner + Muon-3×, seed 3401 (the bake-off's arm, banked) | 1.5146 | −0.0691 |
| our winner + Muon-3×, seed 3402 (this control's second-seed arm) | 1.5027 | −0.0810 |
| **GDN2 + Muon-3×, option A (THE CONTROL)** | **1.4702** | **−0.1135** |
| GDN2 + Muon-3×, option B (the informative convention-mirror appendix) | 1.4151 | −0.1686 |

The pre-stated rule (`muon_control.VERDICT_RULE`, receipted in every arm's
`result.json`): the lead is RETRACTED if GDN2+Muon lands below the banked
winner+Muon final (1.514617647443499). It does, under BOTH routing
conventions. The retraction is robust to the one convention question this
control had (§1): the maximal-Muon routing (A) lands 0.0444 below our row,
and the convention-mirror routing (B) lands 0.0995 below it.

Three honest precisions on that verdict:

* **Under the approved maximal routing the two machines are at PARITY within
  the variance band, not separated.** The endpoint gap A-vs-winner (0.0444)
  sits inside the ~0.075-nat endpoint-variance band the side-by-side
  receipts (`RESULT_SIDE_BY_SIDE_0239.md` §1); at step 160 the two curves
  are within 0.005 (1.4604 vs 1.4649), and their steady-state (steps
  80–160) means vs GDN2's AdamW are −0.023 vs −0.032. What the control
  refutes is not "our Muon arm is worse than GDN2's Muon arm" — it is the
  bake-off's claim of a LEAD. At the matched surface there is no lead in
  either direction under A; under B GDN2 is clearly ahead (steady −0.213,
  crossing GDN2's own AdamW curve at step 40 already).
* **Muon helps GDN2 too — by less than it helped us, which is itself the
  finding.** GDN2's Muon gain over its own AdamW is −0.1135 (A) / −0.1686
  (B); our winner's Muon gain over its own AdamW is −0.3081 / −0.3200 (two
  seeds). The bake-off's C5 re-read said the binding constraint was the
  training surface, not the architecture's ceiling; the control sharpens it:
  the surface was masking most of the 0.239 AdamW-gap (it closes to ~parity
  under A), but the surface does not INVERT the machine ranking once both
  machines get it. GDN2's architecture still holds the best number on the
  box.
* **The narrower claim that survives:** our winner + Muon-3× beats GDN2's
  *banked AdamW row* at both seeds (−0.069, −0.081) — but that is an
  optimizer-for-AdamW comparison, exactly the asymmetry this control was
  built to remove, and it is not the question. The question was whether our
  machine with its best optimizer beats GDN2 with its best optimizer. It
  does not.

## 1. The protocol and the arms (the approved conventions)

Every arm runs the EXACT probe protocol verbatim: the 20,000,000-target
matched-exposure rule (the 20,022,866 landing, 180 steps, every arm
identical), seed 3401 (the seed axis is the second arm's own variable), the
registered flagship stream (identity asserted), the document-disjoint
calibration eval at 7 windows / 14,336 targets (recorded-read path), the
honest clock, the protocol's own grad-clip (1.0, error_if_nonfinite), and
the bake-off's own checkpoint plan (steps 1/5/10/20/40/80/120/160/180; CPU
state snapshots inside the loop, every eval on restored snapshots after
training — the training path unperturbed by construction). GDN2's model
code is UNTOUCHED: `matched_arms.build_gdn2` (the registered lattice_gdn2
arm, its searched nomination's knobs, asserted at launch against the banked
row) is imported verbatim; the Muon is `bakeoff_optimizers.HybridMuonX45`
imported verbatim — the same pinned KellerJordan quintic (`hybrid_muon`
FP32 NS arithmetic, momentum .05-lerp + Nesterov .95, Moonshot
0.2·√max(shape) scaling) the bake-off receipted. No `k10_*`/`k6_*`/bakeoff/
side-by-side source, PREREGISTRATION or `lattice.py` was edited; all new
code is `muon_control.py` + its leased launcher + tests.

**The GDN2 parameter-inventory mapping (option A, supervisor-approved):**
GDN2's inventory is 597 tensors / 149,855,670 params — per block (27
blocks): the gdn2 core's q/k/v/b/w projections, f_proj.0/1 (the decay
pre-activation MLP), g_proj.0/1 (the output SiLU-gate MLP), o_proj, the
SwiGLU mlp w1/w2/w3 (all 2D matrices), plus embed/head, and the non-matrix
params (three RMSNorms + o_norm, A_log/dt_bias — the decay-gate parameters,
g_proj.1.bias, the 3D depthwise q/k/v short-conv kernels). Option A routes
**Muon on every 2D hidden-layer matrix** — embed, head, all ten gdn2
projections, mlp w1/w2/w3: **353 tensors / 149,682,176 params (99.9%)** —
and **AdamW on the true non-matrix params** — 244 tensors / 173,494 params.
The rationale (approved): it matches the directive's own phrasing, and it
delivers the MAXIMAL Muon surface to GDN2 — the conservative direction for
our lead claim (a lead that survives GDN2's maximal-Muon control is receipted
against the strongest possible control). The K10 bake-off exclusion was
about the decay-gate PARAMETER (a small non-matrix-class param), not
gate-producing matrices. Option B (the appendix arm, informative only): the
gate-producing projections (b/w_proj, f_proj.0/1, g_proj.0/1) stay on AdamW
— Muon 191 tensors / 122,255,360 params (81.6%) — mirroring the K10
decay-gate exclusion literally. The inventory is ENUMERATED in code: any
parameter outside the documented set fails loudly
(`split_parameters_gdn2`), and the tests bind the routing to the real
meta-built inventory's counts.

**The LR convention, optimizer-for-optimizer:** 3× GDN2's OWN banked AdamW
LR — 3 × 5.4201855e-4 = 1.6260557e-3 — with GDN2's own betas
(0.9, 0.9392), weight decay (0.20059), warmup 69 on BOTH hybrid groups (the
prior line's both-groups convention, exactly the bake-off's muon3x arm's
own), microbatch 8. The symmetric read: each machine's Muon arm sits at the
same 3×-of-its-own-AdamW-LR point, so the comparison is
optimizer-for-optimizer, not LR-favored either way.

**Protocol-identical starts, receipted:** every GDN2 control arm's step-1
train loss is bit-identical to the banked gdn2 probe's own first row
(5.948033220866492) — asserted in-code. The winner's second-seed arm ran on
the BANKED `optimizer_bakeoff.py` driver, unmodified, `--seed 3402`.

**Receipts:** `muon_control_runs/gdn2_muon3x_seed3401/`,
`muon_control_runs/gdn2_muon3x_gates_adamw_seed3401/` (each: `start.json`
with the full convention block, `training.jsonl` live per-step,
`checkpoints.jsonl`, `result.json` with the curve, walls, gdn2's own gate
telemetry at the final checkpoint — the mechanism-unchanged receipt — and
the Muon update RMS), `bakeoff_runs/muon3x_seed3402/` (the banked driver's
own receipt format), the assemble summary
`receipts/muon_control_summary.json`, and the lane's sources
(`muon_control.py`, `run_muon_control_leased.sh`,
`tests/test_x45_muon_control.py`). Session lambda01, cards 2/3 (the
side-by-side lane's cards 0/1 untouched), RTX 6000 Ada, torch 2.9.1+cu128.
Committed per arm as each landed.

## 2. The overlaid curves (THE artifact)

Calibration eval NLL (nats/byte, 7 windows / 14,336 targets) at matched
checkpoints — the control arms overlaid on the banked bake-off curves (gdn2
and the incumbent are the side-by-side's banked armed runs; winner-muon3x
seed 3401 is the bake-off's banked arm — all cited, unmodified).

| step | MB | gdn2 AdamW (banked) | winner+muon3x s3401 (banked) | winner+muon3x s3402 | **gdn2+muon3x A** | gdn2+muon3x B (appendix) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.7045 | 5.8013 | 5.7726 | 5.8986 | 5.8977 |
| 5 | 0.57 | 3.8999 | 5.6138 | 5.5616 | 5.5076 | 5.4941 |
| 10 | 1.19 | 3.1024 | 5.0646 | 4.9560 | 4.4089 | 4.3434 |
| 20 | 2.37 | 2.6675 | 3.8080 | 3.8099 | 3.1477 | 2.9845 |
| 40 | 4.54 | 2.2309 | 2.7015 | 2.7592 | 2.5227 | 2.0999 |
| 80 | 8.88 | 1.8528 | 1.8427 | 1.9244 | 1.8670 | 1.5686 |
| 120 | 13.28 | 1.6290 | 1.5826 | 1.6236 | 1.5898 | 1.4150 |
| 160 | 17.72 | 1.5056 | 1.4649 | 1.4921 | 1.4604 | 1.3638 |
| 180 | 20.02 | 1.5837 | 1.5146 | 1.5027 | 1.4702 | 1.4151 |

The same table as deficits vs GDN2's banked AdamW curve (negative = better
than GDN2's AdamW row; the bake-off's primary readout):

| step | winner+muon3x s3401 | winner+muon3x s3402 | gdn2+muon3x A | gdn2+muon3x B |
|---:|---:|---:|---:|---:|
| 1 | +0.097 | +0.068 | +0.194 | +0.193 |
| 5 | +1.714 | +1.662 | +1.608 | +1.594 |
| 10 | +1.962 | +1.854 | +1.307 | +1.241 |
| 20 | +1.140 | +1.142 | +0.480 | +0.317 |
| 40 | +0.471 | +0.528 | +0.292 | **−0.131** |
| 80 | **−0.010** | +0.072 | +0.014 | **−0.284** |
| 120 | **−0.046** | −0.005 | −0.039 | **−0.214** |
| 160 | **−0.041** | −0.013 | **−0.045** | **−0.142** |
| 180 | **−0.069** | −0.081 | **−0.113** | **−0.169** |

**What the curves say:**

* **Muon's early hole is Muon-machinery, not our model — it reproduces on a
  completely different architecture.** GDN2+Muon leaves init slowly
  exactly as our winner's Muon arm did (the hole vs GDN2's own AdamW peaks
  at step 5–10: +1.61/+1.31 for A, +1.59/+1.24 for B, vs our winner's
  +1.71/+1.96) and then descends faster than everything else. The
  orthogonalized-momentum buildup interacting with a warmup schedule is a
  property of the hybrid itself, on any machine — the bake-off's
  "no-warmup Muon" next cell is now motivated for both machines.
* **The winner's four-checkpoint "lead over gdn2" was always against
  GDN2's AdamW.** Against GDN2's Muon curve there is no lead at any
  checkpoint: A tracks our winner's muon3x curve within ±0.01 from step 80
  to 160 (1.8670/1.5898/1.4604 vs 1.8427/1.5826/1.4649) and the endpoint
  separates only inside the variance band (1.4702 vs 1.5146); B is ahead of
  our winner's curve from step 40 on, everywhere.
* **GDN2+Muon descends through its own AdamW's curve and keeps going** —
  A crosses GDN2's AdamW at ~step 80–120 (matching our winner's crossing
  step), B at step 40. The endpoint draws: A's 1.4702 and B's 1.4151 both
  sit above their own step-160 values (1.4604/1.3638) — the same
  endpoint-jump noise the ladder receipts on every arm at this protocol;
  no conclusion here rests on the last 0.01–0.05 of any endpoint.

## 3. The readouts per arm

| arm | final NLL | vs winner+muon3x 1.5146 | step-10 hole (vs own machine's AdamW) | steady deficit (steps 80–160 vs gdn2 AdamW) | train s | Muon cost s/step |
|---|---:|---:|---:|---:|---:|---:|
| gdn2+muon3x **A (the control)** | **1.4702** | **−0.0444** | +1.307 | −0.023 | 615.7 | +0.323 |
| gdn2+muon3x B (appendix) | 1.4151 | −0.0995 | +1.241 | −0.213 | 593.0 | +0.189 |
| winner+muon3x seed 3402 | 1.5027 | −0.0119 | +1.854 (s3401: +1.962) | +0.018 (s3401: −0.032) | 1907.4 | +0.176 |

Walls are honest-clock class-matched to the banked rows (GDN2's arms ~3.3
s/step vs its banked AdamW's ~3.0; the winner's seed arm 1907 s vs its
banked 1913 s). Muon's optimizer cost scales with matrix count as the
bake-off receipted (353 matrices: +0.32 s/step; 191: +0.19; the winner's
72: +0.18). The Muon update RMS is healthy and in the same class on every
arm (0.168 / 0.177 / 0.184 / 0.184).

**The mechanism is receipted UNCHANGED in every control arm** (the control's
scope rule: optimizer wiring only). The GDN2 arms' step-1 train loss is
bit-identical to the banked row's (5.948033220866492, asserted in-code), and
GDN2's own gate telemetry at the final checkpoint — read non-invasively from
its own projections, its source untouched — is the banked regime:

| arm | state retention exp(g) | erase gate b | write gate w | implied horizon (tok) |
|---|---:|---:|---:|---:|
| gdn2 AdamW (banked, side-by-side §4) | 0.684 | 0.494 | 0.515 | 2.1–7.4 per layer |
| gdn2+muon3x A | 0.697 | 0.503 | 0.507 | 3.5 mean |
| gdn2+muon3x B | 0.675 | 0.486 | 0.561 | 3.2 mean |

Better weights on the same always-on dense gated-delta machinery — GDN2's
mechanism is untouched and its gates stay in their learned regime; only the
weights moved.

## 4. The second-seed variance, stated honestly

Our winner + Muon-3× at seed 3402 lands at **1.5026696068899972** vs seed
3401's 1.514617647443499 — a seed delta of **−0.0119** (seed 3402 slightly
better). The seed axis varies the model-init draw only: the stream, the
data order and the eval windows are fixed by the registration, so this is a
two-draw read of the init variance, not a data-order variance.

Read against the bake-off's margins:

* **The 0.069 margin over GDN2's banked AdamW row is ~6× the observed seed
  delta and sign-stable at both seeds** (−0.069, −0.081) — the
  winner+Muon-beats-GDN2's-AdamW statement does not owe its existence to
  the single seed. That caveat is now closed.
* **But the seed band does not rescue the lead claim either.** The
  symmetric control's margins are −0.044 (A) and −0.100 (B) against our
  BEST seed (1.5027), and the steady-state read is worse: the winner's
  muon3x steady lead over GDN2's AdamW (−0.032 at seed 3401) FLIPS to
  +0.018 at seed 3402 — the "steady lead over gdn2" itself was inside the
  seed noise all along. What is seed-stable on our machine is Muon's win
  over our own AdamW (−0.31/−0.32 at the two seeds, far above any variance
  band); what is NOT stable is any claim of beating GDN2 at a matched
  surface.

## 5. The honest final answer to the operator's question, stated symmetrically

*What is the best optimizer/schedule for this machine?* — **hybrid Muon at
3× the AdamW LR point, on a short warmup — for BOTH machines.** It is the
best training surface anyone has measured on this box at this protocol: it
improves OUR winner by 0.31–0.32 nats over its tuned AdamW (two seeds) and
GDN2 by 0.11–0.17 over its own searched AdamW. The bake-off's
optimizer-surface conclusion (its §6 recommendation to register hybrid Muon
as the default, with the 2700 s confirmation cell before any scale-up
displaces AdamW) STANDS, now with its single-seed caveat closed on the
winner.

*But the bake-off's lead claim is RETRACTED.* The 0.069 "our winner BEATS
gdn2" was our-best-optimizer vs GDN2's-AdamW. Give GDN2 the same optimizer
surface and it lands at 1.4702 (at the maximal-Muon routing, parity with
our machine inside the variance band) to 1.4151 (at the convention-mirror
routing, clearly ahead). **GDN2 remains ahead: at the matched training
surface the machine ranking the ladder banked (GDN2 first) is restored,
and the residual architecture gap is ~0 at the maximal routing and ~0.10
in GDN2's favor at its best routing.** The right next lever is NOT another
optimizer arm on our machine — the surface is now matched and the gap that
remains is structural (the side-by-side's C5 write/read form, unchanged by
this control; its §6 named lever stands).

*The C5 re-read, re-scoped once more, honestly:* the bake-off said Muon-3x
"closes the steady deficit and inverts it" — the inversion was against
GDN2's AdamW and does not survive the matched surface. What survives and
sharpens: the ~0.16-nat AdamW-era steady deficit was mostly TRAINING
SURFACE — under the matched Muon surface it shrinks to ~parity (A) — so the
side-by-side's original C5 attribution must be read as "the write form
undertrains under AdamW", not "the write form is worth 0.16 nats at any
optimizer". The structural comparison (any next write-form arm) must now
be run under the Muon surface, exactly as the bake-off's §6 rule 2 already
demands.

## 6. Residuals, stated

* **Option B outperformed option A** (1.4151 vs 1.4702) — GDN2's
  gate-producing projections train better under AdamW at GDN2's own LR than
  under Muon at 3×. This is informative in both directions: the
  maximal-Muon routing is NOT GDN2's optimal routing (so the control, if
  anything, UNDER-delivered GDN2's ceiling — the conservative direction for
  our claim, as approved), and by symmetry our winner's Muon routing has an
  untested gates-on-AdamW variant too. No such arm was run (outside the
  approved scope; named here as the natural follow-up cell for the
  convention discussion, not as a claim).
* **Two seeds on the winner, one on GDN2's Muon arms** — the GDN2 control
  arms are seed-3401 single-seed, like every banked row they are compared
  against; the seed band measured on the winner (±0.012) is the only seed
  evidence available and no conclusion here rests on margins smaller than
  it except where stated.
* **Probe scale, one protocol** — everything above is the 20 M probe (180
  steps, seed class 3401/3402, 7-window eval); the 2700 s board and C14+
  scale are untested, as the bake-off already stated for its own numbers.
* **The endpoint-variance band (~0.075 nats)** applies to every endpoint
  number in §0; the verdict's margins (−0.044 A, −0.100 B vs our best seed)
  straddle the band, which is why §0 states the A-read as PARITY-within-
  band rather than a GDN2 win, and why the retraction rests on the
  combination (no lead under A within the band; a clear GDN2 win under B)
  rather than on A's endpoint alone.
* **The 10× Muon point remains untested** on either machine (outside the
  approved bracket, as in the bake-off).

## Commands

```bash
# one control arm, detached on a leased card from the diagnosis lane's pool
setsid bash -c 'OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=2,3 \
  experiments/expressivity_tasks/x45_matrix_emender/run_muon_control_leased.sh \
  gdn2_muon3x > …/muon_control_runs/gdn2_muon3x.console.txt 2>&1' &

# the winner's second-seed arm (the BANKED bake-off driver, unmodified)
setsid bash -c 'OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=2,3 \
  experiments/expressivity_tasks/x45_matrix_emender/run_muon_control_leased.sh \
  muon3x_seed3402 > …/bakeoff_runs/muon3x_seed3402.console.txt 2>&1' &

# the overlay + verdict summary (the control table, the margins, the rule)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.muon_control assemble

# the routing/convention tests (CPU; includes the real-inventory enumeration)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_x45_muon_control.py -q
```
