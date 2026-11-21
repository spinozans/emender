# RESULT — K18 THE SHORT-CONV TRANSPLANT: the missing-component candidate
with published backing, tested on the composed machine, and the answer is
PARITY

**Date:** 2026-11-26 · **The registration:** `PREREGISTRATION_SHORT_CONV.md`
(DRAFT, committed before any probe measurement, 1c97ba9d; the escalated
placement and the two corrected specs approved by the supervisor's ruling
of 2026-11-26). **Receipts:** every number below is from
`k18_runs/<arm>_seed3401/result.json`, `receipts/k18_short_conv_summary.json`
(the assemble) and `receipts/k18_gates.json` (the engine gates). The lane
runs under the operator's EXPLICIT DIRECTIVE (the K17 ruling's own
condition; this document is that directive's answer).

**The question:** GDN2's registered block carries a short depthwise
convolution (conv_size=4, default ON) that our comparison machine has
NEVER had. The delta-rule family's own ablations say the short conv is
load-bearing locally (0.05–13.5 ppl heterogeneous across the lineage).
Is the conv the missing component that owns part of the composed machine's
remaining ~0.03 vs GDN2? **No directional prediction was pre-stated; the
THREE-OUTCOME rule was.**

## 1. The arms' rows (matched-exposure probe, corrected scoring, seed 3401)

| arm (post-cooldown PRIMARY) | post-cooldown | constant-180 | EMA-180 | vs the composed machine's same-seed 1.3811 |
| --- | ---: | ---: | ---: | ---: |
| **the conv transplant** (born identity `[0,0,0,1]`, learns) | **1.3770291294370378** | 1.4312722001756941 | 1.4275548117501395 | **+0.0041** |
| **the frozen-conv falsifier control** (frozen at the identity birth) | **1.3811104467936925** | 1.433674931526184 | 1.424389226096017 | **+0.0000 — BIT-IDENTICAL to the banked row, every digit** |
| **the fixed-shift control** (frozen `[0,0,0.5,0.5]`, never learned) | **1.3820685488837106** | 1.4441970075879778 | 1.4156022923333305 | **−0.0010** (inside the band, the harmful side) |
| the composed machine's own row (the K15 arm, CITED — the baseline) | **1.3811104467936925** | 1.433674931526184 | 1.424389226096017 | — |
| **GDN2 (the gap reference)** | **1.3521734476089478** | | | |

**The bands on every margin:** the 0.075-nat endpoint-variance band and
the winner's 0.0119 seed delta. **No second seed is owed** (the conv arm's
+0.0041 is under the 0.01 win bar; the standing rule applies exactly as
stated). No arm contacted GDN2's row.

## 2. The verdict, per the pre-stated THREE-OUTCOME rule

**SC-2 PARITY.** The conv transplant's +0.0041 is deep inside the 0.0119
seed band: **the conv is NOT the residual's owner on byte-LM text at probe
scale — the uniformity attribution stands with one more receipt.** The
pre-stated small-end possibility is receipted, not excused: the published
with/without effects are heterogeneous (0.05 ppl Kimi Linear to 13.5
DeltaNet LAMBADA), and ours measured at the small end — a real, published
regime, not a measurement failure. No clause conflicted, no
quality-conclusion inversion, **`escalation_flags` EMPTY**.

**The SH-1 decomposition (the shift control's answer, reported as
measured):** ANY LOCAL MIXING alone (the frozen
current+previous average) is **−0.0010** vs the baseline — inside the
band, marginally harmful — while the LEARNED FILTER (the conv arm) is
**+0.0041** — inside the band, marginally helpful — a **+0.0050**
conv-vs-shift separation, both parity-grade. The decomposition's honest
reading at this exposure: a fixed local blur of the payload is mildly
disliked (consistent with the slot semantics' write-what-you-read
structure), and the conv's LEARNED response to that pressure is to stay
essentially at the identity (§3) — the optimizer itself chose
current-token dominance.

**THE FROZEN-CONV FALSIFIER PASSED PERFECTLY** — the lane's strongest
plumbing receipt, matching the K17 falsifier's: the control's step-1 loss
is bit-identical to the composed machine's own row (5.782775391046648),
the landing is bit-identical to the banked row on every scoring
(1.3811104467936925 / 1.433674931526184 / 1.424389226096017), and **the
full 204-step trajectory is BIT-IDENTICAL to the composed machine's own
banked training log (204/204 steps, max per-step |Δloss| = 0.0, no first
divergence)** — the identity-born conv and its frozen optimizer routing
are a NO-OP at the strongest possible bar.

## 3. The mechanism receipts (the direct reads)

* **THE LEARNED KERNELS (the mechanism's own question: did the conv learn
  a token-shift-like mix?): NO — the conv learned to stay AT the
  identity.** The trained per-tap means across the 256 channels:
  **[−0.0005, −0.0007, −0.0011, 0.9645]** at step 180 and
  **[−0.0004, −0.0006, −0.0010, 0.9623]** post-cooldown — the current tap
  at 0.96, the three past taps at −0.001, per-tap spread ~0.017, mean
  distance from the identity birth **0.046/0.048** (max 0.129). The
  published "previous-token/induction" mechanism is NOT what this machine's
  payload path adopted: under the additive read's interface the optimizer
  keeps the current-token payload dominant. The conv moved — it is not a
  dead parameter (the step-0 gradient receipt) — but it moved TOWARD
  identity-preserving territory, exactly the direction that cannot move
  NLL beyond parity.
* **The decorative-store sweep (the uniformity hypothesis's direct test):
  UNCHANGED.** Layers ≥ 0.02 dNLL: **4 of 24 post-cooldown** (5 of 24 at
  step 180) — the banked regime's own 4–5. The local temporal filter did
  NOT make more layers load-bearing. The uniformity attribution stands.
* **The C5 instrument on the conved payload: the composed machine's own
  within noise** — effective rows 15.89 (the composed machine's own 16.83),
  the top-row collapse detector 0.164 (0.155), occupancy 26.27 (27.20),
  the identity-rewrite share 0.105 (0.127). No collapse, no regime change.
* **The gate telemetry: UNMOVED** — all three p3 channels at the composed
  machine's own scales (decay drift 0.0108 vs 0.0107, write 0.0240 vs
  0.0233, read 0.0119 vs 0.0113 post-cooldown). The conv touched neither
  the gates nor the decisions, and the receipts say so.
* **The honest walls:** training 1932.7 s constant (the composed arm's
  own 1920.8 — the conv costs +0.6 % at the probe protocol, matching gate
  D's 0.75 % block-call receipt); peak memory 28.07 GB.

## 4. Ours-best, quoted TWO WAYS (the K17 ruling's own discipline — the
band makes them indistinguishable, and this document must not let the
better-looking number silently become "the" number)

* **(a) The BANKED two-seed best — unchanged: the composed machine's own
  1.3811104467936925 / 1.383232252938407 pair** (the citation-grade row;
  the falsifier here re-receipts it bit-for-bit).
* **(b) The BEST ABSOLUTE ROW — the conv arm's 1.3770291294370378**,
  SINGLE-SEED, parity-grade over its own baseline (+0.0041, inside the
  band, no second seed owed) — edging the K17 dense arm's 1.377752116748265
  by +0.0007, a margin the band cannot rank.

The remaining-gap range across all of them: **+0.0249 (the conv arm) to
+0.0310 vs GDN2's 1.3521734476089478 — sign-stable, INSIDE the 0.075-nat
endpoint-variance band. THE BAND CANNOT RANK THEM.** The conv arm's row
puts the program's best absolute at 69.6 % of the corrected gap closed —
a single-seed, parity-grade row, reported as measured.

## 5. The X46 disclosure (the supervisor's addition, stated for the record)

**The machine that beat the Welch-bound ceiling in X46 (gdn2-fixed 0.93 vs
0.80) is the conv-HAVING machine.** GDN2's registered block carries the
short conv ON (conv_size=4, default) — the ceiling refutation was achieved
WITH the conv present and is not conv-attributable-in-doubt: the linear
ceiling was derived from the key-space geometry regardless of the conv,
and the trained machine exceeded it with ALL its components, conv
included. This lane's parity result does not weaken that receipt, and
the receipt does not contradict this lane's: a component can be
load-bearing for a multi-byte-key binding task (X46's separator) while
being small-end on byte-LM text at probe exposure.

## 6. The honest scope of this result

* Probe scale only (the 20 M matched exposure, 180+24 steps, 7-window
  eval), seed 3401 (the pre-stated win-bar rule owed no second seed
  anywhere); single-protocol rows.
* The conv arm ADDS parameters (+24,576 = +0.015 %) — negligible, quoted.
* The silu mismatch (ours never silus the payload; GDN2 silus q/k/v even
  with the conv OFF) and the q/k-conv absence are standing, disclosed
  cross-machine differences this lane did NOT touch: a conv+silu arm and
  the q/k-conv variant (P2 re-registered as decisions conv-influenced)
  are the NAMED FOLLOW-UPS. This lane's SC-2 result is for the
  TEMPORAL-FILTER object on the payload path — the same object GDN2's own
  ablation isolates.
* The composed-machine-vs-GDN2 row is a cross-machine context row at the
  same probe protocol, not a controlled comparison.
* The engine's receipts (`K18_SHORT_CONV.md`, `receipts/k18_gates.json`):
  born as the composed machine 0 flips in 516,734,976 lanes at probe
  scale, the fp64 gradcheck through the conv path green, the kernel
  gradients through the conv chain at 0.0 deviation, the committed
  K10/K13/K11B/K15/K17 suites 111/111 with the module loaded, the lane's
  own suite 25/25 + the probe tests 7/7.

## 7. Commands

```bash
# the arms (the detached leased driver; cards 0/1, one card per arm):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k18_short_conv_leased.sh conv 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k18_runs/\
conv_seed3401.console.txt 2>&1 < /dev/null &

# the assemble + the three-outcome adjudication:
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv assemble

# the lane's tests (CPU) + the engine suite (CUDA):
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_k18_short_conv_probe.py -q
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  experiments/expressivity_tasks/x45_matrix_emender/test_k18_short_conv.py -q
```
