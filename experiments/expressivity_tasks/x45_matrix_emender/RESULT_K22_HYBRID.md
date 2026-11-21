# K22 — THE HYBRID: the result

**Status: landed. Both arms at seed 3401, matched exposure (20,022,866
targets, the registered flagship stream), the corrected cooldown-primary
scoring, the banked GDN2 row's own recipe verbatim (HybridMuonX45 5x,
option B, warmup 69, microbatch 8). Registration draft committed before
any arm (`PREREGISTRATION_K22_HYBRID.md`). Engines imported verbatim, never
modified. Cards 2/3 only.**

## The headline, as plainly as the evidence supports

**The falsifier passed perfectly — the strongest plumbing receipt in the
program's hybrid series.** The frozen-at-zero-read hybrid is bit-identical
to the banked GDN2 row's own trajectory at **every one of its 204 steps**
(max per-step delta **0.0**), lands at the banked **1.3521734476089478**
post-cooldown bit for bit, matches the banked constant-180
(1.3950541530336653) and EMA-180 (1.3803383282252721) digit for digit, and
reproduces the banked step-1 train loss (5.948033220866492). Born as pure
GDN2, bit for bit, forever while the gates stay closed — the K18
perfect-receipt precedent beaten at a harder seam (a whole foreign engine
grafted into the committed stack).

**H-1 is REFUTED at parity grade — the literature's minority-interleave
gain did NOT transfer to byte-LM at 150M/20M.** The live hybrid (the gates
learn to open) lands at **1.3547673906598772** post-cooldown vs pure GDN2's
**1.3521734476089478**: +0.0026, **deep inside the 0.0119 winner-seed
band** — parity phrasing per the pre-stated band discipline (no
band-worship, and no overclaim of "hurts" either: the delta is inside
within-machine seed noise). The pre-registered REFUTATION clause fires:
receipted plainly, **the uniformity finding EXTENDS to the hybrid regime**.
The secondary scorings agree: constant-180 1.4082818201610021 vs GDN2's
1.3950541530336653 (+0.0132), EMA-180 1.3816877944128854 vs 1.3803383282252721
(+0.0014). **No second seed is owed** (no >= 0.01 win; the standing rule).

**H-2 is HELD — the Meta complementarity receipt transfers.** The hybrid
lands **below the composed machine's own same-seed row by 0.0263**
(1.3547673906598772 vs 1.3811104467936925 at 3401; 1.383232252938407 at
3402) — outside the winner-seed band: **the slot store pays inside a
dense stack even though the pure slot machine lost**. GDN2's stack with
three slot layers beats our best pure machine by ~0.026 nats/byte; it just
doesn't beat GDN2 itself.

## The table (post-cooldown primary, both parents' banked rows cited)

| arm | post-cooldown | constant-180 | EMA-180 | step-1 | wall (const) |
|---|---|---|---|---|---|
| **GDN2 (banked, cited)** | **1.3521734476089478** | 1.3950541530336653 | 1.3803383282252721 | 5.948033220866492 | 620.2 s |
| **hybrid_frozen (falsifier)** | **1.3521734476089478** (bit-identical) | 1.3950541530336653 (bit-identical) | 1.3803383282252721 (bit-identical) | 5.948033220866492 (bit-identical) | 912.7 s |
| **hybrid (live)** | **1.3547673906598772** | 1.4082818201610021 | 1.3816877944128854 | 5.948033220866492 | 919.4 s |
| composed-ours (banked 4x, cited) | 1.3811104467936925 / 1.383232252938407 | 1.433674931526184 | 1.424389226096017 | 5.782775391046648 | 1,932.7 s (K18 arm) |

Both live scores' step-1 is bit-identical to GDN2's banked row (the
protocol-identical start receipt: the machines ARE pure GDN2 at birth).
The honest wall: the hybrid trains the same 20,022,866 bytes in 1.48x
GDN2's wall-clock (919.4 s vs 620.2 s; the K20 projection said 1.37x at
3-of-24, and the committed block checkpointing re-runs each slot scan in
backward — the projection's own caveat, receipted).

## THE DECORATIVE SWEEP — the headline instrument: the uniformity finding
does NOT invert

The slot-read mask per slot layer, post-cooldown, at the trained open
gates (|gate| ~ 0.13):

| masked slot layer | dNLL |
|---|---|
| 6 (gate -0.130) | **-0.00030** |
| 12 (gate +0.133) | **-0.00055** |
| 18 (gate +0.133) | **+0.00100** |
| all three | **-0.00009** |

**Zero of three slot layers clear the 0.02 load-bearing bar** (the pure
machine's own banked regime is 4-5 of 24 layers >= 0.02 dNLL). The slot
reads are DECORATIVE inside the dense stack — the uniformity finding
extends to the hybrid regime, now with its most direct receipt: even when
the model is free to read the store (all three gates open, monotonically,
to |gate| ~ 0.13 by step 180), removing every slot read changes the
post-cooldown NLL by **-0.00009** — nothing. The literature's matched-param
minority-interleave gains (Jamba/Samba/Meta) do not appear here; at this
scale, in this stack, the dense mixers already supply what the slot store
would.

## The mechanism receipts (the C5 instrument on the slot layers)

* **The gate trajectories**: all three read gates open monotonically from
  exact zero — step 1 |g|~4e-5, step 20 ~0.0075, step 40 ~0.026, step 80
  ~0.052, step 120 ~0.080, step 160 ~0.112, step 180 ~0.121, final
  ~0.132 (layer 6 NEGATIVE: -0.130; layers 12/18 positive: +0.133). The
  open signal receipt at birth: dL/dgate = -154.18 on the fixture, and
  every other slot parameter's gradient exactly zero while a gate is
  closed. The gates learn to read the store; the store's content just
  doesn't matter to the loss.
* **The op-mix rows inside the dense stack** (post-cooldown, per slot
  layer): CORRECT-dominant 97.2-98.3%, APPEND 1.6-2.6%, MERGE ~0.08-0.16%,
  POP never — the same qualitative regime as the pure machine (97.4%
  correct / 2.3% append / 0.26% merge at the banked anchor).
* **Occupancy / effective rows / identity share vs the pure machine's**:
  effective read rows **25.26** (the pure machine's own 15.9-16.8),
  occupancy **35.09** live slots (the pure machine's 26.2-33.5), identity
  rewrite share **0.0555** (the pure machine's 0.118-0.126 — half). Inside
  the dense stack the stores read a more diverse row distribution and
  rewrite identity half as much — the store runs differently, not less.

## The gate receipts (before any arm; `receipts/k22_gates.json`)

Gate A — born as GDN2, bit for bit: the full-model loss and every slot
layer's combined output bit-identical to a pure-GDN2 twin's own on one
real production microbatch, **0 flips in 31,457,280 lanes** (the IEEE
sign-of-zero corner did not fire once). Gate A2 — the seam's dtype
discipline holds (the production bf16 residual stream stays bf16 through
the seam). Gate B — the fp64 gradcheck through the hybrid seam (the
slot-in-dense-stack boundary) green in both gated combinations; the kernel
path's seam gradients within the measured fp32-vs-fp64 floors. Gate C —
the open signal nonzero at birth, every other slot gradient exactly zero,
the store fills (append 230 / correct 794 on the fixture). Gate D — the
slot branch's own forward wall 20.96 ms per (slot layer, window-group) at
the production shape (the K20 receipt's 23.40 ms at width 576, same
regime).

## The falsifier's own receipt, stated fully

`hybrid_frozen`: the whole slot branch `requires_grad=False` (51 tensors),
outside the optimizer, the grad clip and the EMA by construction; the
parameter split then equals the banked GDN2 row's own exactly (191 Muon /
406 AdamW tensors, verified against a pure-GDN2 twin's own split). Result:
**204/204 steps bit-identical to
`protocol_correction_runs/gdn2_speedrun_rescored/training.jsonl`**, max
per-step delta 0.0, every scored row the banked row's own digits. The
attribution is as clean as it can exist: the live arm's +0.0026 belongs to
the slot reads opening — and the sweep says those reads carry ~nothing.

## Adjudication and standing state

* The pre-stated clauses, applied exactly as registered: H-1 REFUTED at
  parity grade (+0.0026, inside the 0.0119 seed band — the parity
  phrasing); H-2 HELD (0.0263, outside the seed band); the REFUTATION
  clause receipted plainly (the uniformity finding extends to the hybrid
  regime, per the sweep the direct receipt); no second seed owed; the
  falsifier PASSED PERFECTLY; escalation flags EMPTY (no birth-parity
  conflict, no standing conclusion inverted — the refutation was itself
  the pre-registered outcome).
* The banking of any row and the final gap-quote phrasing are the
  supervisor's to adjudicate (the K13 clause-conflict precedent). This
  document states the measured rows and the flags.
* The lane's velocity note (the K20/K21/K23 receipts): at matched TIME the
  hybrid sees ~68% of GDN2's exposure (919.4 s vs 620.2 s per
  matched-exposure landing at the probe scale) — the hybrid's quality case
  would need to absorb that concession at board scale, and this lane's
  result says it has no quality surplus to absorb it with. The shared-pool
  variant and the K23 batched-launch integration (3.32x bit-identical
  per-layer-step scan pair, projected 87.8% exposure at 3-of-24) remain
  the named follow-ups if the hybrid program continues.
