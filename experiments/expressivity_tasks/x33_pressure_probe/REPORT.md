# X33 — the pressure-predicts-collapse probe: REPORT

**Eval-only.** Two bounded X28E tape arms rebuilt from the hash-frozen package
(read-only; freeze manifest `0b674c71…aa11b4` re-verified by the probe at run
start, 185 files), selected checkpoints loaded from the sibling results
directory, immutable ladder pools re-evaluated with per-token routing traces.
No training, no optimizer, no writes outside this directory. All bin
accuracies reproduce the X28E `final_eval.json` receipts **byte-identically**
(8/8 crosscheck gates passed; see `data/run_log.txt`).

- `tape_B0` (budget **B = 736** slots/head), best.pt sha256 `29ad3a96…31f146`
  (selected update 5000). Traced words: 6,400 (4 bins × 512 + 17 sweep × 256).
- `tape_B0x4` (budget **B = 2944**), best.pt sha256 `ced743eb…e744e1`
  (selected update 7000). Traced words: 10,752 (4 bins × 512 + 34 sweep × 256:
  its own window **and** B0's sweep points k=608..864 for the same-k
  comparison; sweep pools are seed-fixed functions of k, identical across
  arms).
- Walls (leased card 5; cards 2/3/4 at 99–100% util with other experiments,
  card 7 excluded): B0 22.6 s, B0x4 1307.6 s. Extractor self-test (vectorized
  pressure vs brute-force oracle, adversarial PAD/undefined-window cases):
  PASS, 24/24 checks.

## 1. The mechanism (measured, single-word routing dump, B0 k=864 accept)

All 4 heads run in **lockstep**. Per token: SOS → R-EMPTY (1 allocation);
then the word alternates **R-CONFLICT** (adjacency-addressed correction, no
allocation) and **R-NOMATCH** (append) on *every* head — through the a-phase
**and** the b-phase (the model never stops writing during b's). POP never
fires (n_pop = 0 in **every** traced word of both arms). Consequences:

- Allocations grow ~1 per 2 tokens per head: the word's total append demand
  at depth k is **~k+2 slots** (not ~k: the b-phase keeps appending).
- The store fills at the *fixed absolute position* **t ≈ 2B−2** (B0: token
  1470 at k=864; B0x4: ~5886), independent of k.
- After fill, the alternation collapses: **every token on every head fires
  R-CAP** (t=1472..EOS in the dump), live pinned at B to EOS.
- At the boundary k=736 exactly the final b-append + EOS degrade (8 R-CAPs
  per word); the count channel loses its last b and every balanced word is
  rejected — collapse is total (accept recall 0.0) at k=736 while k=720 is
  perfect.

## 2. Accuracy-vs-k and pressure-vs-k (fine sweeps)

`cumP` = mean cumulative R-CAP fraction over accept words (head-pooled, within
word); `SAT` = fraction of accept words with a W64 saturation event (trailing
64-token window ≥ 0.9); `first_sat` = median first saturation token index.

### tape_B0 (B=736) — the cliff is exactly the budget

| k | acc | cumP | SAT | first_sat | peak live |
|---|---|---|---|---|---|
| 608–720 (8 pts) | **1.0000** | **0.0000** | 0.0 | — | k+1 (<736) |
| 736 | 0.5000 | 0.0027 | 0.0 | — | 736.0 |
| 752 | 0.5000 | 0.0442 | 0.0 | — | 736.0 |
| 768 | 0.5000 | 0.0823 | 1.0 | 1524 | 736.0 |
| 784 | 0.5000 | 0.1175 | 1.0 | 1524 | 736.0 |
| 800 | 0.5000 | 0.1501 | 1.0 | 1524 | 736.0 |
| 816 | 0.5000 | 0.1804 | 1.0 | 1524 | 736.0 |
| 832 | 0.5000 | 0.2086 | 1.0 | 1524 | 736.0 |
| 848 | 0.5000 | 0.2349 | 1.0 | 1524 | 736.0 |
| 864 | 0.5000 | 0.2596 | 1.0 | 1524 | 736.0 |

Bins: ID 0.9941 / 2x 0.9805 / 4x 0.9844 (cumP 0.0000 everywhere), **9x
0.5000** with cumP 0.4407 (k=1025) → 0.5312 (k=1152), SAT 1.0, first_sat
1524, peak pinned 736. Per-shape at 9x: balanced 0.00; leading_b /
count_mismatch / trailing_a all 1.00 — collapse = "never accepts".

### tape_B0x4 (B=2944) — own window

| k | acc | cumP | SAT | first_sat | note |
|---|---|---|---|---|---|
| 2816–2928 (8 pts) | 0.8320 | 0.0000 | 0.0 | — | peak k+21 < B; count_mismatch-only failures |
| 2944 | 0.8320 | 0.0007 | 0.0 | — | peak pinned B; **first R-CAP** |
| 2960 | 0.8320 | 0.0114 | 0.0 | — | |
| 2976–3024 (4 pts) | 0.8320 | 0.02–0.05 | 1.0 | 5940 | saturation, still 0.832 |
| 3040–3072 (3 pts) | **0.5000** | 0.06–0.08 | 1.0 | 5940 | collapse |

The 0.832 plateau is **entirely count_mismatch failures** (j=k±1 accepted;
balanced/leading_b/trailing_a all 1.0 at k=2816): an off-by-one count
blindness that pressure does not see (cumP 0.0000 through k=2928). Pressure
onset (k=2944) **leads** accept-collapse (k=3040) by 96 depth; windowed
saturation (k=2976) leads it by 64 depth.

### The same-k comparison (B0's sweep points, both arms)

| k | B0 acc | B0 cumP | B0x4 acc | B0x4 cumP |
|---|---|---|---|---|
| 608–720 | 1.0000 | 0.0000 | **0.5000** | 0.0000 |
| 736–816 | 0.5000 | 0.003–0.18 | **0.5000** | 0.0000 |
| 832–864 | 0.5000 | 0.21–0.26 | **1.0000** | 0.0000 |

**B0x4 collapses (acc 0.5, always-reject) at k=608..816 with pressure exactly
0.0000 and peak live ≈ k+1 ≤ 849 ≪ 2944, and holds again at 832–864.**
Independently confirmed by a direct minimal forward (no probe code): accept
margins −0.003 (reject) at k=560..816 vs +0.000..0.001 (accept) at 832–864 —
a near-flat verdict margin flipping sign. The hole spans at least [560, 816]
(bin_4x ≤512 holds 0.9707; bin_9x 1025–1152 holds 0.9473) — a **non-capacity
collapse band** invisible to the coarse registered bins.

## 3. Word-level contingency (accept words; "fired" = any R-CAP in word)

| arm | accepts | fired | fired→incorrect | incorrect total | PPV | sensitivity |
|---|---|---|---|---|---|---|
| tape_B0 | 3200 | 1408 | **1408 (100%)** | 1408 | **1.000** | **1.000** |
| tape_B0x4 (all) | 5376 | 1152 | 384 | 2181 | 0.333 | 0.176 |
| tape_B0x4 excl. hole band+ bins (own sweep) | 1152 | 1152 | 384 | 384 | 0.333 | **1.000** |

At B0 the raw pressure register separates perfectly: every pressured word is
wrong, every wrong word is pressured, zero false alarms on 1,792 correct
accepts. At B0x4 the 768 fired-but-correct words are exactly the k=2944..3024
leading edge (pressure fired, collapse 64–96 depth ahead); the 1,792
unflagged failures are the hole band (+5 bin_4x accept misses).

## 4. The coincidence test (pre-registered landmarks)

| landmark | tape_B0 | tape_B0x4 (own sweep) |
|---|---|---|
| k_acc99 (first acc < 0.99) | 736 | 2816 |
| k_acc75 (first acc < 0.75) | **736** | **3040** |
| first k with any R-CAP | **736** | **2944** |
| k_p50 (cumP ≥ 0.5) | 1103 (bin_9x) | none (max 0.081) |
| k_psat (cumP ≥ 0.9) | **none** (max 0.531) | none |

- B0: accuracy falls below 0.75 **exactly where pressure first fires**
  (k=736; k=720 holds with zero pressure). Cumulative pressure exceeds 0.5
  only at k=1103 — *inside* the 9x band, ~367 depth **after** collapse — and
  never reaches 0.9 (max 0.5312 at k=1152). The windowed register first
  saturates at k=768, two sweep steps after collapse onset.
- B0x4 at the same k values (736–864): pressure 0.0000 (well below
  saturation) — but accuracy is 0.5 there too (the hole), so the implied
  "B0x4 holds where B0 collapses" contrast is **false** at 736–816 and true
  only at 832–864 and in the 9x band (0.9473, pressure 0.0000).

## 5. Lead/lag of the register (pre-registered P6, as measured)

Saturation fires at the fixed budget-geometry position **first_sat ≈ 2B+52**
(B0: 1524; B0x4: 5940) — **inside the b-phase** (lead_b negative: median
−707 at B0 k=864-band; −2883 at B0x4), *not* before the first b token as
pre-registered. It still precedes the **EOS verdict** by:

- B0 sweep collapsing words: lead_eos = 13 tokens (k=768) … 205 (k=864),
  exactly 2k−1471, deterministic;
- B0 bin_9x accepts: 527–781 tokens (median 655);
- B0x4 collapsing sweep words: 141–205 tokens (median 173).

So the observable fires while the word is still streaming, before the
decision position — but after count evidence has begun arriving, not before
it.

## 6. Verdict against the pre-registration (RECEIPTS.md)

- **H1 (the directed hypothesis): MIXED**, exactly the pre-registered mixed
  case: cumulative pressure rises steeply and monotonically past the budget
  (0.0027 → 0.2596 over k=736–864; 0.44–0.53 in the 9x band) but **never
  saturates (≥0.9) at any probed depth**, while the windowed register does
  saturate (1.0) at every probed k ≥ 768.
- **The honest null fires twice**: (a) accuracy collapses without pressure
  saturation — B0x4's hole band (k∈[560,816], 1,792 accept words, pressure
  identically 0, occupancy ≤849/2944); (b) pressure saturation without
  (registered-criterion) collapse — B0x4 k=2976..3024 saturates at accuracy
  0.832 (though the count channel is already ±1-blind there and full
  accept-collapse follows within 64 depth).
- P1: **partial** — zero events at k ≤ 720 ✓; events on 100% of collapsing
  accept words at k ≥ 768 ✓; but k=736/752 collapse with no W64 event
  (raw R-CAP does fire: 8 and 136 per word).
- P2: **mixed** — steep rise ✓, k_p50 = 1103 ∈ [736, 1152] ✓, cumulative
  ≥ 0.9 at collapse ✗ (max 0.53).
- P3: **refuted as stated** — the pressure half holds (0.0000, zero events,
  at all 17 shared k), but B0x4 does *not* hold accuracy at B0's collapse
  depths: it collapses at 608–816 too.
- P4: **supported** — 9x band: accuracy 0.9473 ≥ 0.9, cumP 0.0000 < 0.5, no
  sustained saturation.
- P5: **supported** — first events at k=2944, exactly where the word's
  append demand (~k+2) crosses the 2944 budget; collapse (k=3040) coincides
  with saturation in the sense that saturation onset precedes it by 64–96
  depth and every collapsing word is saturated.
- P6: **refuted as registered, reformulated** — saturation is mid-b-phase
  (lead_b < 0), not pre-b; it precedes the EOS verdict by 13–781 tokens.

## 7. What can honestly be claimed (the citable form)

1. **The pressure register is a perfect budget-exhaustion detector at B0**:
   across 6,400 traced words, any R-CAP fire ⟺ the word's accept verdict
   fails (PPV 1.000, sensitivity 1.000, zero false alarms), and the
   collapse cliff sits at exactly the depth where the learned solution's
   ~k+2-slot demand crosses B=736.
2. **It leads collapse in depth at B0x4**: pressure onset at k=2944
   (budget), accept-collapse at k=3040 — a 96-depth early-warning band in
   which 768 balanced words are still answered correctly but flagged; every
   word that eventually collapses is saturated first.
3. **It is a *specific*, not *sensitive*, predictor**: a large collapse band
   (B0x4, k∈[560,816]) occurs with pressure identically zero and occupancy
   3.5× under budget — a non-capacity failure the register correctly does
   not blame on capacity. Pressure saturation is evidence *about capacity
   collapse only*; its absence is not evidence of health.
4. **Cumulative R-CAP fraction is a graded dose-response curve, not a
   binary saturating signal** (0.001→0.53 over the collapse range); the
   windowed register is the saturating observable, and it saturates at a
   fixed geometric position 2B+52 tokens into the word, 13–781 tokens
   before the verdict.

## 8. Artifacts

- `data/tape_{B0,B0x4}_per_example.jsonl` — per-example scatter (17,152
  words: k, label, pred, shape, cumP per head, first_sat, sust_sat,
  peak/eos live, lead_b/lead_eos).
- `data/tape_{B0,B0x4}_per_k.json` — per-k aggregates; `_case_counts.json` —
  per-pool routing-case counts; `_series_examples.json` — per-token
  req/rcap series; `_run_receipt.json` — freeze/checkpoint identity, pool
  hashes, walls; `run_log.txt` — full crosscheck log.
- `probe.py` (instrument + self-test oracle), `analyze.py` (all tables
  above), `run_all.sh` (lease + smoke + full run).
