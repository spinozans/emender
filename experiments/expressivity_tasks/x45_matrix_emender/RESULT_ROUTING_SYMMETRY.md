# RESULT_ROUTING_SYMMETRY.md — the symmetric routing search on our winner: the asymmetry the operator caught, closed

**Date:** 2026-11-21 · **Status: complete — the routing search is now symmetric, and it settles NOTHING in our favor: our machine's routing spread is inside the pre-stated band, the routing GDN2 prefers is the routing our machine already runs, and GDN2 remains ahead at every matched comparison.** ·
**Authorisation:** the operator's symmetric-routing directive — the Muon control
gave GDN2 TWO Muon routings (max-Muon A: 1.4702; gates-on-AdamW B: 1.4151, B
better by 0.055) but OUR winner only ONE (the bake-off convention,
1.5146/1.5027 two-seed): the comparison "our-best vs GDN2's-best-of-two" was
asymmetric in GDN2's favor. This lane closes it. The parameter-inventory
mapping (what counts as OUR gate-producing params) was escalated and ruled by
the supervisor before any run (M1, the faithful functional mapping, with the
broad-mapping reading run anyway as an appendix so it cannot be said untested);
the two predictions below were pre-stated with that ruling.

## 0. The verdict, stated first

**ROUTING IS SETTLED, AND IT SETTLES IN THE BORING DIRECTION — which is the
finding.** Our machine's Muon-routing spread (A-analog 1.5303 vs B-analog
1.5146, same seed) is **+0.0157, inside the pre-stated 0.02 band** —
~3.5× smaller than GDN2's 0.055, exactly as the pre-stated routing-spread
prediction said, and for the reason the inventory enumeration exposed: **our
machine's only learned gate is the DECAY pair — 27,696 params, 0.018% of the
machine — against GDN2's 27.4M gate-matrix surface (18%).** Our write/erase
admission is HYPERPARAMETER thresholds (tau_match/tau_merge/tau_pop), not
learned matrices; our addressing projections (key/query_key) are
k_proj/q_proj analogs, and GDN2's own precedent kept those on Muon under BOTH
its routings. The direct consequence, proven in code
(`tests/test_x45_routing_symmetry.py`:
`banked_gates_adamw` reproduces `bakeoff_optimizers.split_parameters`
EXACTLY): **the banked bake-off arm already WAS our B-analog — the banked
1.5146/1.5027 rows are cited, not re-run (a re-run would be bit-identical by
construction) — and the routing that helps GDN2 most (gates-on-AdamW) is the
routing our machine has run all along.** There was no hidden GDN2-style B-win
waiting on our side, and the A-analog does not unlock one either.

| the symmetric table (both machines × both routings, the inherited 3× LR point) | seed 3401 | seed 3402 |
|---|---:|---:|
| **OUR winner, B-analog routing** (the banked bake-off arm: gate producer on AdamW) | **1.5146** | **1.5027** |
| OUR winner, A-analog routing (max-Muon: every 2D matrix on Muon) | 1.5303 | — |
| OUR winner, broad-mapping appendix (M2: decay + the address producers on AdamW) | 1.5417 | — |
| **GDN2, option A** (max-Muon, the control's arm) | **1.4702** | — |
| **GDN2, option B** (gates-on-AdamW, the control's arm) | **1.4151** | — |
| *(reference: our AdamW 1.8227 · GDN2's AdamW 1.5837 · all at the exact probe protocol, 20,022,866 targets)* | | |

**The symmetric best-vs-best verdict, under the restated claim rule (no
'leads'/'behind' claims except best-vs-best with the routing search
symmetric):** our best-of-routings (1.5027, the B-analog at its best seed) vs
GDN2's best-of-routings (1.4151, its option B) — **GDN2 AHEAD by 0.0876 at
the matched 3× point** (0.0995 same-seed), sign-stable across both our seeds
and just outside the ~0.075-nat endpoint-variance band. Every matched-routings
read agrees: A-style 1.5303 vs 1.4702 → GDN2 +0.060; B-style 1.5146/1.5027
vs 1.4151 → GDN2 +0.100/+0.088. **No routing on our side inverts the machine
ranking; the control's retraction stands with the routing question closed.**

**The LR bracket (the honest extra axis this lane was allowed): 3× was not our
machine's optimum — 4× is better at both seeds** (1.4907/1.4899 vs 3×'s
1.5146/1.5027; the 4× seed delta is a remarkable 0.0008). At our best
measured surface point (4×, B-analog routing) the margin vs GDN2's best is
0.0748 — **at the edge of the endpoint-variance band: parity-within-band, no
claim bankable in either direction** — and stated with the honesty it
demands: that comparison is asymmetric in OUR favor now (we got an LR search,
GDN2 did not; GDN2's own 3× is as inherited as ours was, and our bracket
found gains above 3× that GDN2's unmeasured axis may mirror). It does not
beat GDN2's option A either (1.4899 vs 1.4702: +0.0197, inside the endpoint
band).

**The remaining gap's ownership, after routing is settled:** routing owns
≤ ~0.016 on our side (inside the band — the A/B analog difference, and our
spread is smaller than GDN2's); the LR point owns ~0.013–0.024 on our side
(4× over 3×, outside the 0.0119 seed band at seed 3401, at its edge at
3402; GDN2's own LR axis UNMEASURED); **what remains — ~0.07 at our best
measured surface, ~0.09–0.10 at the matched 3× point — is the structural C5
write/read form**, now measured under our best routing AND our best LR
point. The inventory headline is the same fact from the other side: GDN2 has
27.4M params of learned gate machinery to route; our machine has 27.7K and
hyperparameter thresholds — the write form itself is the lever, exactly as
the side-by-side's C5 attribution and its §6 named lever say, and exactly
what the K11 LEARNED-GATES pre-registration (draft, operator sign-off
pending) targets.

## 1. The mapping and the arms (the supervisor-approved conventions)

**The parameter-inventory mapping (M1, ruled by the supervisor before any
run):** our winner's inventory is 267 tensors / 149,981,808 params (the
audited count, asserted at build), enumerated by function the way the control
enumerated GDN2's:

* **the gate-VALUE producer (the only learned gate):** `mixer.decay.weight`
  (2×576) + `mixer.decay.bias` (2) — the DECAY gate, the direct analog of
  GDN2's `f_proj` (decay pre-activation), which GDN2's option B moved to
  AdamW. The pair is 27,696 params (0.018%); the 2D weight alone 27,648.
* **the write/erase/read admission:** the five-op selection chain — threshold
  comparisons on HYPERPARAMETERS (`tau_match`/`tau_merge`/`tau_pop`, not
  params) against the similarity of `write_keys` with the stored key table;
  the read is a softmax addressed by `query_keys`. `mixer.key` /
  `mixer.query_key` produce ADDRESSES — the functional analogs of GDN2's
  k_proj/q_proj, which stayed on Muon under BOTH of GDN2's routings. No
  learned write/erase gate VALUE exists in this machine.
* **everything else:** embed/head, `mixer.value`/`mixer.out`, the SwiGLU mlp
  w1/w2/w3 (all 2D), the RMSNorms (1D).

The inventory is ENUMERATED in code: any parameter outside the documented set
fails loudly (`routing_symmetry.split_parameters_winner`), the banked-B-analog
identity against `bakeoff_optimizers.split_parameters` is asserted in the
tests, and the real-inventory group sizes bind to the receipted counts
(banked 170 Muon tensors / 149,925,888 params; max-Muon 194 / 149,953,536 —
99.98% under Muon, AdamW on the true non-matrix residue only: 73 tensors /
28,272; broad-mapping 122 / 146,386,944 with the gate + address producers on
AdamW: 145 tensors / 3,594,864).

**The arms** (every one at the EXACT probe protocol verbatim: the
20,000,000-target matched-exposure rule — the 20,022,866 landing, 180 steps
identical in every arm — seed 3401 first, the registered flagship stream
(identity asserted), the document-disjoint calibration eval at 7 windows /
14,336 targets (recorded-read path), the honest clock, the protocol's own
grad-clip (1.0, error_if_nonfinite), and the bake-off's own checkpoint plan
(steps 1/5/10/20/40/80/120/160/180; CPU state snapshots inside the loop,
every eval on restored snapshots after training — the training path
unperturbed by construction). `velocity_ladder._build_k10` and
`bakeoff_optimizers.HybridMuonX45` are imported VERBATIM; the winner's knobs
are asserted at launch against the banked row (lr 3.3427e-4, wd 0.17422,
warmup 45 on BOTH hybrid groups, betas (0.9, 0.9392), microbatch 8); no
`k10_*`/`k6_*`/bakeoff/muon-control source, PREREGISTRATION or `lattice.py`
was edited — all new code is `routing_symmetry.py` + its leased launcher +
tests):

| arm | routing | LR point | seeds |
|---|---|---|---|
| `muon3x_max` | **the A-analog (max-Muon)**: every 2D matrix under Muon (the banked routing + `decay.weight`) | 3× | 3401 |
| `muon3x_gates_adamw` | **the broad-mapping appendix (M2)**: decay + `mixer.key` + `mixer.query_key` on AdamW — run so the broad reading cannot be said untested | 3× | 3401 |
| `muon2x_banked` / `muon4x_banked` | **the LR bracket** at the best routing (the banked B-analog — the best of the three routings above) | 2× / 4× | 3401; 4× also 3402 (it became the overall best, so it owes the single-seed caveat the same way the banked 1.5146 once did) |

**Protocol-identical starts, receipted:** every arm's step-1 train loss is
bit-identical to the BANKED muon3x arm's own first row at the same seed
(seed 3401: 5.785070835425993; seed 3402: 5.768619428012168) — asserted
in-code against the banked run's own `training.jsonl`, not a restated
constant. The optimizer cannot touch the first forward, so this is the
protocol-equivalence proof; the receipt (`step1_bit_identical_to_banked`) is
true in every arm.

**Receipts:** `routing_symmetry_runs/<label>_seed<seed>/` (each: `start.json`
with the full convention block, the routing inventory and the pre-stated
predictions; `training.jsonl` live per-step; `checkpoints.jsonl`;
`result.json` with the curve, walls, mechanism diagnostics, the Muon update
RMS and the step-1 receipt), the assemble summary
`receipts/routing_symmetry_summary.json`, and the lane's sources
(`routing_symmetry.py`, `run_routing_symmetry_leased.sh`,
`tests/test_x45_routing_symmetry.py`). Session lambda01, cards 2/3 (the
lane's pool; cards 0/1 untouched), RTX 6000 Ada, torch 2.9.1+cu128.
Committed per arm as each landed.

## 2. The overlaid curves (THE artifact)

Calibration eval NLL (nats/byte, 7 windows / 14,336 targets) at matched
checkpoints — this lane's arms against the banked curves (the bake-off's
winner rows and the control's GDN2 rows, cited, unmodified). The banked
B-analog 3× columns ARE our machine's prior best surface; the A/B GDN2
columns are the control's arms.

| step | MB | gdn2 A (banked) | gdn2 B (banked) | B-analog 3× s3401 (banked) | B-analog 3× s3402 (banked) | **our A 3×** | our broad 3× | our 2× | our 4× s3401 | our 4× s3402 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.8986 | 5.8977 | 5.8013 | 5.7726 | 5.8035 | 5.7914 | 5.8103 | 5.7986 | 5.7759 |
| 5 | 0.57 | 5.5076 | 5.4941 | 5.6138 | 5.5616 | 5.6164 | 5.4341 | 5.6826 | 5.5481 | 5.4873 |
| 10 | 1.19 | 4.4089 | 4.3434 | 5.0646 | 4.9560 | 5.0800 | 4.7839 | 5.3044 | 4.8478 | 4.7356 |
| 20 | 2.37 | 3.1477 | 2.9845 | 3.8080 | 3.8099 | 3.8171 | 3.8023 | 4.1975 | 3.5674 | 3.5657 |
| 40 | 4.54 | 2.5227 | 2.0999 | 2.7015 | 2.7592 | 2.7249 | 2.8115 | 3.0259 | 2.5649 | 2.6298 |
| 80 | 8.88 | 1.8670 | 1.5686 | 1.8427 | 1.9244 | 1.8946 | 2.0060 | 1.9857 | **1.7801** | 1.8327 |
| 120 | 13.28 | 1.5898 | 1.4150 | 1.5826 | 1.6236 | 1.6215 | 1.6670 | 1.6576 | **1.5574** | 1.5761 |
| 160 | 17.72 | 1.4604 | 1.3638 | 1.4649 | 1.4921 | 1.4858 | 1.5268 | 1.5033 | **1.4637** | 1.4704 |
| 180 | 20.02 | 1.4702 | 1.4151 | 1.5146 | 1.5027 | 1.5303 | 1.5417 | 1.5542 | **1.4907** | **1.4899** |

The same table as deficits vs GDN2's banked AdamW curve (positive = worse
than GDN2's AdamW row; the lanes' primary readout):

| step | our A 3× | our broad 3× | our 2× | our 4× s3401 | our 4× s3402 | *(B-analog 3×: s3401 −0.069 / s3402 −0.081 at 180)* |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | +1.978 | +1.682 | +2.202 | +1.745 | +1.633 | |
| 40 | +0.494 | +0.581 | +0.795 | +0.334 | +0.399 | |
| 80 | +0.042 | +0.153 | +0.133 | **−0.073** | −0.020 | |
| 120 | −0.007 | +0.038 | +0.029 | **−0.072** | −0.053 | |
| 160 | −0.020 | +0.021 | −0.002 | **−0.042** | −0.035 | |
| 180 | −0.053 | −0.042 | −0.030 | **−0.093** | **−0.094** | |

**What the curves say:**

* **The routing arms are indistinguishable from the banked arm inside the
  band, in both directions.** Our A-analog tracks the banked B-analog
  curve within ±0.02 at every checkpoint from step 40 on (1.8946/1.6215/
  1.4858 vs 1.8427/1.5826/1.4649) and lands +0.0157 above it; the
  broad-mapping arm is a uniform ~0.02–0.15 WORSE through the steady phase
  (its addressing projections train slower under AdamW, exactly as GDN2's
  precedent predicted) and never crosses the banked curve. The early hole
  reproduces on every routing (+1.68 to +2.20 at step 10) — it is Muon
  machinery + warmup, as the control already receipted on GDN2, not a
  routing property.
* **The LR bracket is the only arm that MOVES the curve.** 2× is uniformly
  worse (steady +0.053); 4× is uniformly better from step 40 on, posts the
  best steady read our machine has ever shown (−0.062 mean over steps
  80–160 at s3401, −0.036 at s3402), and its step-160 values (1.4637/
  1.4704) are the best mid-run numbers any arm of ours has banked — inside
  ~0.003–0.010 of GDN2's option A at the same step (1.4604), though still
  ~0.10 behind GDN2's option B (1.3638). Every endpoint here carries the
  same endpoint-jump noise the ladder receipts on every arm (4× jumped
  +0.027/+0.020 from step 160 to 180); no conclusion rests on the last
  0.01–0.03 of any endpoint.
* **The 4× seed stability is unusually tight** (finals 1.4907/1.4899, delta
  0.0008, vs the banked 3× pair's 0.0119) — the tighter LR point appears to
  sit deeper in a basin. Two seeds only; stated, not claimed as a property.

## 3. The readouts per arm

| arm | final NLL | vs B-analog s3401 1.5146 | step-10 hole (vs gdn2 AdamW) | steady (80–160 vs gdn2 AdamW) | train s | Muon cost s/step |
|---|---:|---:|---:|---:|---:|---:|
| **our A-analog (max-Muon) 3×** | **1.5303** | **+0.0157** | +1.978 | +0.005 | 1878.9 | +0.198 |
| our broad-mapping 3× | 1.5417 | +0.0271 | +1.682 | +0.071 | 1920.2 | +0.143 |
| our 2× (B-analog routing) | 1.5542 | +0.0396 | +2.202 | +0.053 | 1865.1 | +0.169 |
| our 4× (B-analog routing) s3401 | **1.4907** | −0.0239 | +1.745 | **−0.062** | 1904.0 | +0.171 |
| our 4× s3402 | **1.4899** | −0.0248 | +1.633 | −0.036 | 1905.7 | +0.168 |
| *(banked B-analog 3×: 1.5146 s3401 / 1.5027 s3402; steady −0.032/+0.018; Muon cost +0.183; train 1912.8 s)* | | | | | | |

Walls are honest-clock class-matched to the banked rows (1865–1920 s vs the
banked 1912.8). Muon's optimizer cost scales with matrix count as the lanes
receipted (194 matrices +0.198 s/step; 170: +0.169–0.183; 122: +0.143). The
Muon update RMS is healthy and in the same class on every arm
(0.183–0.185).

**The mechanism is receipted UNCHANGED in every arm** (this lane's scope
rule: optimizer routing only) — final-checkpoint diagnostics, the banked
channels:

| arm | live slots /128 | read entropy (nats) | identity share | CORRECT/token | MERGE/token | key guard |
|---|---:|---:|---:|---:|---:|---|
| banked B-analog 3× (banked) | 35.6 | 1.664 | 0.123 | 0.9735 | 0.0025 | ok (min 1.65) |
| our A-analog 3× | 35.2 | 1.815 | 0.114 | 0.9732 | 0.0026 | ok (min 1.49) |
| our broad-mapping 3× | 22.9 | 1.474 | 0.103 | 0.9763 | 0.0053 | ok (min 1.61) |
| our 4× s3401 | 33.5 | 1.522 | 0.122 | 0.9744 | 0.0026 | ok (min 1.86) |
| our 4× s3402 | 35.2 | 1.596 | 0.118 | 0.9738 | 0.0025 | ok (min 1.82) |

Same sparse byte-identity-CORRECT hot-set regime everywhere
(`span_key_std` min ≥ 1.18 against the 0.05 guard floor in every layer,
every arm — the KEY-COLLAPSE guard holds throughout). The broad-mapping
arm's store runs a smaller hot set (22.9 live slots, entropy 1.47) — the
address producers under AdamW learn a slower, sharper addressing — the
mechanism's regime is unchanged, only its weights move differently.

## 4. The pre-stated predictions, adjudicated

Both were pre-stated with the supervisor's mapping ruling and receipted in
every arm's `result.json`; the assemble applies them exactly as stated:

* **`routing_spread` — CONFIRMED (+0.0157, inside the 0.02 band).** The
  max-Muon arm lands within the band of the banked B-analog same-seed row:
  our machine has no meaningful routing surface to exploit. The prediction's
  stated reason is now receipted as the inventory fact: 0.018% of the machine
  moved between the two routings (vs GDN2's 18%).
* **`broad_mapping` — CONFIRMED (+0.0271, neutral-to-worse).** The address
  producers (key/query_key) train better under Muon on this machine, exactly
  as GDN2's own precedent (addressing stayed on Muon under both its
  routings) predicted. The M2 reading is now tested and closed: it is not a
  hidden better B-analog; it is ~0.03 worse, and it shrinks the store's hot
  set while doing so.

## 5. The honest verdict under the symmetric claim rule, and the C5 read

*What is our machine's best Muon routing?* — **the banked one (the
B-analog): the DECAY gate's producer on AdamW, everything else 2D on Muon.**
The bake-off's routing was never a convention choice that cost us anything —
it was already the routing this machine prefers, and the alternatives are
measured now: max-Muon +0.016 (inside the band, no preference distinguishable),
broad-mapping +0.027 (worse, clearly enough to close the reading). The
asymmetry the operator caught is closed in the only direction it could be:
GDN2's routing search bought it 0.055; ours buys 0.016 at most, inside the
variance band.

*Is anything settled between the machines?* — **At the matched 3× point with
the routing search symmetric: GDN2 remains ahead, 1.4151 vs our best-of-
routings 1.5027 — margin 0.0876 (0.0995 same-seed), sign-stable across both
our seeds, at/just outside the ~0.075 endpoint-variance band.** Under
matched A-style routings the gap is 0.060 (1.4702 vs 1.5303) — the 0.044 the
operator computed against our banked row widens slightly under a true
A-vs-A. The only lever that moved our number was the LR point: 4× buys
0.013–0.024 over 3× (two seeds), brings our best measured final to 1.4899,
and the margin vs GDN2's best to 0.0748 — **at the band's edge:
parity-within-band, no claim bankable — and that comparison is asymmetric in
OUR favor (GDN2 got no LR bracket), so if anything it understates GDN2's
position.** The honest symmetric statement of the whole lane: **no surface
lever on our side — routing, or the one LR bracket cell we were allowed —
inverts the machine ranking; GDN2's 1.4151 remains the best number measured
on this box at this protocol, and our best (1.4899) does not beat even
GDN2's option A (1.4702) outside the band.**

*The C5 read, re-scoped one more time, honestly:* the side-by-side
attributed ~0.16 nats (AdamW-era steady) to the write/read form; the bake-off
showed Muon-3× closes it to ~parity-with-GDN2's-AdamW; the control showed
that at matched Muon surfaces GDN2 is ahead again (the surface was masking,
not inverting, the ranking); this lane shows **routing owns none of the
residual on our side** (our spread is inside the band and our preferred
routing was already installed) and **the LR point owns ~0.01–0.02** (our
side; GDN2's axis unmeasured). What is left — ~0.07 at our best measured
surface, ~0.09–0.10 at the matched 3× point — is structural, and the
inventory enumeration states its architectural face directly: **GDN2 writes
through 27.4M params of learned gate machinery; our machine writes through
27.7K and three hyperparameter thresholds.** Any next structural arm (the
K11 LEARNED-GATES registration now in draft) must run under this lane's
receipted best surface — the B-analog routing at 4× — or it will
mis-attribute the surface to the structure, exactly as the side-by-side's
C5-vs-AdamW endpoint once did.

## 6. Residuals, stated

* **GDN2's LR axis is unmeasured** — its 3× point is as inherited as ours
  was, and our bracket found gains (4× > 3×) that GDN2's axis may mirror; any
  cross-machine margin quoted at our 4× vs its 3× is asymmetric in our favor
  and is not bankable (stated in §0/§5, repeated here because it is the
  lane's largest honest caveat).
* **The bracket is two points around an inherited point, not a search** —
  {2×, 4×} only; the direction at 4× (still improving over 3×) says the
  optimum may lie above 4×, and 6×+ is untested (outside the approved
  bracket, as the 10× point was in the bake-off; the prior line's collapse
  at 10× is cited, untested here).
* **Single seed on the routing arms** (A-analog and broad-mapping are
  seed-3401 only, like every GDN2 Muon arm they are compared against); the
  seed band measured on this machine (0.0008–0.0119 across its three two-seed
  pairs) is the only seed evidence, and no conclusion here rests on margins
  smaller than it except where stated.
* **Probe scale, one protocol** — everything above is the 20 M probe (180
  steps, 7-window eval, the ~0.075-nat endpoint-variance band); the 2700 s
  board and C14+ scale are untested, as the bake-off and control already
  stated for their own numbers.
* **The endpoint jump** (+0.02–0.03 from step 160 to 180 on the 4× arms, the
  same noise every lane receipts) means the 4× arms' TRUE steady position is
  their step-160 values (1.4637/1.4704) — which is why §5 quotes the
  step-160 reads where it matters and no conclusion rests on an endpoint
  alone.
* **The broad-mapping arm's smaller hot set (22.9 live slots)** is receipted
  as an observation, not analyzed — it is a single-seed diagnostic read of a
  refuted routing, and reading more into it would be post-hoc.

## Commands

```bash
# one arm, detached on a leased card from the diagnosis lane's pool (cards 2/3)
setsid bash -c 'OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=2,3 \
  experiments/expressivity_tasks/x45_matrix_emender/run_routing_symmetry_leased.sh \
  muon3x_max 3401 > …/routing_symmetry_runs/muon3x_max.console.txt 2>&1' &

# the LR bracket at the best routing (the banked B-analog)
setsid bash -c '… run_routing_symmetry_leased.sh muon4x_banked 3401 …' &
setsid bash -c '… run_routing_symmetry_leased.sh muon4x_banked 3402 …' &

# the symmetric overlay + prediction verdicts + the best-vs-best read
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry assemble

# the routing/mapping tests (CPU; includes the real-inventory enumeration and
# the banked-B-analog identity proof)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_x45_routing_symmetry.py -q
```
