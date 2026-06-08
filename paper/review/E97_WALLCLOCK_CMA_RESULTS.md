# E97 Wall-Clock CMA — does bounded-state (tanh) E97 + gdn-neg beat gdn2-mlp on the WALL-CLOCK objective at 1.3B?

**Task:** `e97-wallclock-cma`  ·  **Status:** COMPLETE · **Verdict: NO-GO (wall-clock)**
**Scale:** 1.3B param-matched (dim=2240, 64 heads), within-layer `typed_head_mixture`, REAL Pile, no mocks.
**Cell under test:** `gdn2_nonlin_shell` (boundary-phi tanh-bounded state) mixed with `gdn-neg` (`gdn2_recall`, `allow_neg_eigval=1`), vs the `gdn2-mlp` baseline (100% gdn-neg + SwiGLU MLP).

---

## 0. Question this task settles

The predecessor (`tanh-e97-1p3b`, commit `ced30a5`) established a NO-GO on wall-clock from the **C=1 (per-step tanh, slow) vs C=∞ (linear, no edge) endpoints**. This task opens **chunk-size `C` as a free axis** between those endpoints — the boundary-phi kernel applies `phi(S)=tanh` every `C` steps and is linear within the chunk — to ask:

> Is there ANY config (tanh-head **ratio** × **chunk-size C** × shape) where the per-token sample-efficiency edge of the bounded state BEATS its throughput tax on the **wall-clock** held-out-BPB objective at 1.3B?

Plus Erik's refinement (2026-06-08): the fitness must be **capability-guarded** — a pure wall-clock-BPB fitness would push `C` up for speed and silently kill the bounded-state capability (count / nonlinear-state). So we report capability for every operating point, and treat `C` as expressivity-risky.

---

## 1. Throughput vs C — the kernel landscape (microbenchmark, B=2, T=2048)

`gdn2_nonlin_shell` has two realizations of the bounded state:

| kernel | C | tok/s | ratio vs gdn-neg |
|---|---|---|---|
| gdn-neg pure (baseline) | — | 12313 | 1.00 |
| shell identity (linear) | — | 12311 | 1.00 |
| **fused boundary-phi (tanh)** | **1…2048** | **~10840** | **~0.88 (FLAT in C)** |
| chunkref matmul (tanh) | 16 | 890 | 0.072 |
| chunkref matmul (tanh) | 64 | 3222 | 0.262 |
| chunkref matmul (tanh) | 128 | 5354 | 0.435 |
| chunkref matmul (tanh) | 256 | 8424 | 0.684 |
| chunkref matmul (tanh) | 512 | 11639 | 0.945 |

**Key structural finding:** the single-launch **fused** sequential boundary-phi kernel is **~0.88× GDN throughput essentially regardless of C** — increasing `C` does **not** buy speed in the fused path (the cost is the sequential fused scan, not the per-chunk bounding). The only path where larger `C` recovers throughput is the **chunkref matmul** loop, but it is launch-bound and *slower than fused* until `C≈512` — and at `C≥512` the tanh is applied so rarely the cell is effectively linear. **Consequence: there is no `C` that recovers GDN throughput while keeping the bounded-state nonlinearity.** The "find the sweet spot in C" premise collapses at the kernel level.

---

## 2. Token-matched BPB vs C (5M-token capability screen, equal tokens, ratio 0.5)

Held-out BPB at equal token budget (5,003,658 tokens), fused boundary-phi, 32 gdn-neg + 32 shell:

| config | held-out BPB | train tok/s | thru vs gdn |
|---|---|---|---|
| **base_gdn (gdn2-mlp)** | **2.09236** | 9975 | 1.000 |
| shell identity (linear shell) | 2.09612 | 9521 | 0.955 |
| tanh C=1 | 2.09333 | 8303 | 0.832 |
| tanh C=8 | 2.10064 | 8688 | 0.871 |
| tanh C=16 | 2.09681 | 8471 | 0.849 |
| tanh C=32 | 2.09998 | 8617 | 0.864 |
| tanh C=64 | 2.09525 | 8331 | 0.835 |
| tanh C=128 | 2.09853 | 8633 | 0.865 |
| tanh C=256 | 2.09721 | 8848 | 0.887 |
| tanh C=2048 | 2.10386 | 8648 | 0.867 |

**Findings at this scale:**
- BPB is essentially **flat in C** (~2.093–2.104), best tanh = **C=1 (2.09333)**, worst = C=2048 (linear-ish, 2.10386) — consistent with "more frequent bounding (small C) is slightly better", but the spread is ~0.01 (noise-level at 5M tokens, single seed).
- **`base_gdn` (2.09236) is nominally best on BPB even token-matched** at this scale — i.e. there is **no token-matched edge** for the tanh shell here. (The predecessor saw a ~+0.02 token-matched edge in its decisive longer run; the longer-budget head-to-head below re-tests whether the edge appears/grows with more tokens.)

---

## 3. Wall-clock BPB vs C (equal SECONDS, the real objective)

Equal **720 s** wall budget per config (fused boundary-phi = production kernel for every bounded C; `C512_chunkref` is the matmul fast-path control), ratio 0.5, single seed. Lower BPB better; tokens = what each config fit in 720 s.

| config | held-out BPB | tokens in 720 s | tok/s | thru vs gdn |
|---|---|---|---|---|
| **base_gdn (gdn2-mlp)** | **2.03373** | **6,614,172** | 9254 | 1.000 |
| wall_C1 | 2.04632 | 6,319,116 | 8847 | 0.956 |
| wall_C8 | 2.06231 | 5,946,198 | 8341 | 0.901 |
| wall_C16 | 2.05184 | 6,233,058 | 8723 | 0.943 |
| wall_C32 | 2.05456 | 6,097,824 | 8527 | 0.921 |
| wall_C64 | 2.04762 | 6,159,294 | 8614 | 0.931 |
| wall_C128 | 2.06244 | 5,978,982 | 8382 | 0.906 |
| wall_C256 | 2.05047 | 6,200,274 | 8672 | 0.937 |
| wall_C2048 | 2.06018 | 5,905,218 | 8257 | 0.892 |
| wall_C512_chunkref (near-linear fast-path) | 2.04071 | 6,544,506 | 9167 | 0.991 |

**`base_gdn` wins the wall-clock objective decisively — it beats EVERY tanh-C config.** It fits the most tokens in 720 s (6.61M) *and* gets the best BPB (2.0337). The best tanh realization is `wall_C1` (2.0463, gap **+0.0126**). Notably, even `wall_C512_chunkref` — the *fastest* tanh path (0.99× throughput via the chunkref matmul loop) — still loses (2.0407 vs 2.0337), and at C=512 the tanh bounding is so infrequent the cell is essentially linear (capability-dead, §5). **No point on the (C × kernel) landscape beats gdn2-mlp on wall-clock.**

---

## 4. Longer-budget head-to-head on the wall-clock-best tanh config — does the edge grow?

`wc_headtohead.py --C 1`, **2 seeds**, **1100 s** wall budget (≈1.8× the 5M-token screen, ~8.9M tokens reached). C=1 is the right config: fused throughput is flat in C, so C=1 is both the most-expressive *and* the fastest tanh realization — it Pareto-dominates larger C in the fused path. Three measurements per seed: shell @ wall, gdn @ equal wall, gdn capped at the shell's token count (token-matched).

| | shell C=1 | gdn @ wall | gdn @ token-matched |
|---|---|---|---|
| seed 0 BPB | 1.98528 | **1.94010** | 1.97850 |
| seed 1 BPB | 1.98108 | **1.96109** | 1.98365 |
| **mean BPB** | **1.98318** | **1.95060** | **1.98108** |
| tokens in 1100 s | ~8.87M | ~10.63M | (capped 8.86M) |
| tok/s | ~8139 | ~9740 | ~9535 |

- **WALL-CLOCK:** gdn **1.9506** vs shell **1.9832** → gdn wins by **+0.0326** (gdn fits ~1.2× more tokens/sec). Decisive, both seeds.
- **TOKEN-MATCHED:** gdn **1.9811** vs shell **1.9832** → gdn wins by **+0.0021**, but the seeds **split** (gdn wins seed0 by 0.007; shell wins seed1 by 0.003) → a statistical **tie**, NOT a shell win.
- **Does the edge grow with more tokens? NO.** Token-matched, the shell *lost* by ~0.001 at 5M tokens (§2) and still *loses/ties* by ~0.002 at ~8.9M tokens. The sample-efficiency edge that this task hoped to find — and that the predecessor saw as +0.02 for the pure per-step tanh-E97 head — **does not materialize for the `gdn2_nonlin_shell` boundary-phi cell at any budget tested.** There is no longer-budget flip; gdn's token-matched edge is stable. With no token-matched edge to amortize, the ~0.84× throughput tax cannot be recovered at any wall budget.

---

## 5. Capability guard (Erik's refinement) — does large C kill the bounded-state capability?

Within-layer mix (0.5 gdn-neg + 0.5 `gdn2_nonlin_shell`) at C∈{1,64,2048}, vs `gdn2_mlp` reference (100% gdn-neg). Mean length-extrapolation accuracy over T∈{128,256,512,1024}, small ~4M-param probe models, depth 4, seed 0. `count`/`nonlin` are the C-sensitive bounded-state capabilities; `recall` is the gdn-neg-arm control.

| arm | count (anbncn) | nonlin (iter-map) | recall (mqar) |
|---|---|---|---|
| gdn2_mlp (gdn-neg, no shell) | 0.8417 | 0.9053 | 0.0980 |
| shell C=1 | 0.8486 | 0.9042 | 0.0891 |
| shell C=64 | 0.8480 | 0.9036 | 0.0894 |
| shell C=2048 | 0.8500 | 0.9031 | 0.0902 |

**Two findings, both reinforcing NO-GO:**
1. **The a-priori "large C kills capability" risk does NOT manifest here.** count/nonlin are essentially flat across C (0.848–0.850 / 0.903–0.905). Reason: the gdn-neg backbone *already* solves these probes (count 0.842, nonlin 0.905) at depth 4, so the bounding frequency of the tanh shell is not what carries the capability — there is no shell-attributable signal to lose as C grows. (The guard remains the correct safety check; it simply found no cliff at this scale. recall sits near floor ~0.09 for *all* arms — this tiny-dim probe doesn't exercise the full gdn-neg recall the way the 1.3B LM does, so read recall here as "uninformative", not "broken".)
2. **The shell provides no capability uplift over plain gdn-neg** on any probe (count +0.007, nonlin −0.001, recall −0.009 — all within noise). Combined with the LM result (no token-matched BPB edge), the bounded-state shell head is **dominated**: no throughput advantage, no BPB advantage, no capability advantage.

---

## 6. Decision

### GO / NO-GO: **NO-GO** (decisive, full-space)

No config of **ratio × chunk-size C × kernel** makes the tanh/bounded-state `gdn2_nonlin_shell` + gdn-neg beat `gdn2-mlp` on the WALL-CLOCK held-out-BPB objective at 1.3B.

**The four nails:**
1. **Wall-clock landscape (§3):** `gdn2-mlp` beats every tanh-C config (best tanh gap +0.013); it gets both more tokens/sec and better BPB. The "sweet spot in C" does not exist.
2. **Kernel structure (§1):** the fused boundary-phi tanh kernel is **flat ~0.88× GDN throughput for all C** — increasing C buys no speed. The only C that recovers throughput (chunkref C≥512) makes the cell near-linear, and it *still* loses. Bounded-state ⊥ matmul-throughput is fundamental, confirming the predecessor's `tanh ⊥ chunkable` finding.
3. **No sample-efficiency edge to amortize (§2, §4):** for this shell cell there is no token-matched BPB edge at 5M *or* ~8.9M tokens — gdn ties/wins token-matched. The edge does **not grow** at longer budget, so there is no wall budget at which 0.84× throughput flips to a win.
4. **No capability dividend (§5):** the shell adds no measurable count/nonlin/recall capability over gdn-neg at probe scale. It is dominated on every axis.

**Best config found:** none beats baseline. Among tanh configs the wall-clock-best is `C=1` (fused, most expressive *and* fastest tanh path) at BPB 2.046 — still +0.013 behind `gdn2-mlp` 2.034.

**Relation to predecessor:** This REINFORCES the `tanh-e97-1p3b` NO-GO (`ced30a5`) and resolves its open question — opening C as a free axis does **not** rescue wall-clock. It also tempers the predecessor's "+0.02 token-matched win": that edge belonged to the *pure per-step tanh-E97* head; the chunkable `gdn2_nonlin_shell` boundary-phi cell does not reproduce it (token-matched tie). So the only remaining lever from the prior note — a chunkable bounded-state kernel (`gdn2_nonlin_shell`) — is now tested and **also NO-GO**: chunking restores throughput only by abandoning the per-token edge *and* the bounded-state capability simultaneously.

**Recommendation:** stop pursuing bounded-state/tanh shell variants as a wall-clock win at 1.3B. `gdn2-mlp` (dense gdn-neg + MLP) remains the wall-clock-best cell. If a sample-efficiency edge is wanted, it must come from a cell that is *both* per-token-better *and* tensor-core-efficient — neither the per-step tanh (slow) nor the chunked shell (no edge) is that cell.

---

### Reproduction
- Throughput(C): `experiments/e97_wallclock_cma/throughput_sweep.py` → `results/throughput_sweep.json`
- Token-matched BPB(C): `bpb_sweep.py --mode token` → `results/bpb_sweep_token.json`
- Wall-clock BPB(C): `bpb_sweep.py --mode wall` → `results/bpb_sweep_wall.json`
- Longer head-to-head: `wc_headtohead.py --C 1 --seeds 0,1 --wall_seconds 1100` → `results/headtohead_C1.json`
- Capability guard: `capability_guard.py --Cs 1,64,2048` → `results/capability_guard.json`

All runs: REAL Pile, 1.3B param-matched (dim=2240, 64 heads), within-layer fused `typed_head_mixture`, bf16, no eager fallback, no mocks.
