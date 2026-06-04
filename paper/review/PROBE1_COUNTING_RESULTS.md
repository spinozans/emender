# PROBE 1 — Counting-with-comparison: the linear-vs-nonlinear (additive-counter) separator

**Task:** `probe1-counting` · **Model:** claude:opus · **Type:** task build +
positive-control baseline + training run. **`paper/main.typ` was NOT edited.**

Design source: `paper/review/LINEAR_VS_NONLINEAR_TASK_DESIGN.md` §3 (Probe 1).
The motivating result this probe responds to: on the $S_5$ word problem the
*linear* recurrence `e88-linear` **won** (`review/S5_CONFIG_FLIP.md`), because
$S_5$ is finite-state and a reflection-rich input-dependent linear recurrence
reaches it. A real linear-vs-nonlinear separator must therefore live *outside*
the finite-state regime. **Unbounded counting with a comparison is that regime**
[Weiss, Goldberg & Yahav 2018; Délétang et al. 2022; Sarrof–Veitsman–Hahn 2024].

---

## 1. What was built (real code, registered, committed)

### 1.1 Task — `experiments/expressivity_tasks/tasks/counting_with_comparison.py`

Two real counter-language tasks with dense per-position supervision and
length-extrapolation eval (harness contract: `generate_batch(B,T,rng) ->
(input[B,T], target[B,T], mask[B,T])`, `random_baseline_acc()`):

- **`dyck_depth` (Variant 1b, PRIMARY).** Stream over `{'('=0, ')'=1}`; target
  at each position is the current Dyck-1 nesting depth, a non-negative counter
  with a floor:
  `depth_t = max(depth_{t-1} + (+1 if '(' else -1), 0)`.
  The `max(·,0)` floor IS the load-bearing comparison (a `)` at depth 0 is a
  no-op, not a negative count). Depth is displayed capped to `[0, cap=15]` for
  the cross-entropy loss while the *underlying* count is unbounded. A slight
  negative drift (`p_open=0.45`) makes the walk positive-recurrent near the
  floor, so the zero-test fires a constant fraction of the time at **every**
  length — length extrapolation thus measures pure counting stamina, not a
  drifting target distribution. `vocab=16`, `random_baseline≈0.182`
  (always-predict-0, the stationary majority class).

- **`anbncn_viability` (Variant 1a).** Stream over `{a,b,c}`; per-position
  target is the exact online answer to "is this prefix still a viable prefix of
  some $a^n b^n c^n$?", decided by COUNT comparisons (`nb<=na`, `nb==na` to open
  the c-phase, `nc<=na`). The decisive flip (e.g. $a^n b^{n+1}$ becoming
  non-viable at the $(n+1)$-th `b`) sits deep in the sequence and requires
  having counted $n$, so $n\!\sim\!T/3$ forces large-magnitude counting at long
  $T$. Generation mixes valid blocks, off-by-one/perturbed-count blocks, and
  random streams; labels are computed by the exact checker, so every label is
  real. `vocab=3`, `random_baseline=0.5`.

### 1.2 Positive-control baseline — `ndm/models/counter_baseline.py`

The harness's strong "nonlinear" arms (`e88-tanh`, `m2rnn`, `gdn`) are all
saturating- or linear-state, and **tanh is a known false negative for counting**
(WGY 2018). Added two REAL additive / non-saturating counter layers, registered
as ladder levels `relu_rnn` and `lstm`:

- **`relu_rnn`** — additive ReLU-Elman RNN (`torch.nn.RNN(nonlinearity='relu')`),
  `h_t = relu(W_x x_t + W_h h_{t-1} + b)`.
- **`lstm`** — standard LSTM (`torch.nn.LSTM`), additive cell-state accumulation.

Both are the WGY-2018 positive controls: finite-precision ReLU-RNN / LSTM
implement **unbounded counters**; tanh-RNN / GRU and bounded linear-state
recurrences cannot. Wrapped with the standard in_proj/out_proj and the
`(out, h_final)` ladder-layer contract.

---

## 2. Run protocol

- **Arms (param-matched ~8M, same recipe):**
  `e88-linear` (LINEAR, eigenvalue-rich; `linear_state=1`),
  `e88-tanh` (saturating nonlinear; `linear_state=0`),
  `gdn` (`fla-gdn`, gated delta-net, linear state),
  `m2rnn` (matrix-memory RNN),
  `relu_rnn` and `lstm` (additive counters, positive controls).
- **3 seeds** {42,123,456}. **Train T=128.** **Eval T ∈ {128,256,512,1024}**
  (Délétang length-extrapolation protocol).
- schedule-free AdamW, lr 3e-4, batch 32, 3000 steps, depth 4, **fp32
  (`--disable_autocast`)** for all arms — fair across arms and removes the
  bf16 precision confound the design flagged. (Benchmarked: fp32 is not slower
  than bf16 for E88; Triton does not change the recurrence-bound throughput.)
- Param counts (actual): e88-* 7.92M, gdn 8.26M, m2rnn 8.06M, relu_rnn 7.95M,
  lstm 8.05M.

Runner: `experiments/expressivity_tasks/run_probe1_counting.py` (dynamic
idle-GPU scheduler, started on GPUs 2,3). Aggregator:
`aggregate_probe1.py`. Raw per-seed JSONs: `results/probe1_dyck_depth__*.json`.

---

## 3. Results

### 3.1 Variant 1b — `dyck_depth` (running Dyck-1 depth, cap=15, p_open=0.45)

Accuracy at each eval length, mean±std over seeds {42,123,456}. Train T=128.
Random baseline = 0.182 (always-predict-0). Raw JSONs:
`results/probe1_dyck_depth__<arm>__seed<seed>.json`.

| arm         | kind                                 | params | T=128 | T=256 | T=512 | T=1024 |
|-------------|--------------------------------------|--------|-------|-------|-------|--------|
| e88-linear  | LINEAR (eigenvalue-rich)             | 7.92M  | 0.999 | 0.987 | 0.963 | **0.924** |
| e88-tanh    | nonlinear-SATURATING (tanh)          | 7.92M  | 0.999 | 0.978 | 0.951 | **0.929** |
| gdn         | LINEAR (gated delta-net)             | 8.26M  | 0.998 | 0.983 | 0.963 | 0.950 |
| m2rnn       | matrix-memory RNN                    | 8.06M  | 0.997 | 0.959 | 0.863 | 0.731 |
| **relu_rnn**| **ADDITIVE counter (ReLU-Elman)** [WGY+] | 7.95M | 1.000 | 0.990 | 0.971 | **0.961** |
| **lstm**    | **ADDITIVE counter (LSTM)** [WGY+]   | 8.05M  | 1.000 | 0.998 | 0.984 | **0.975** |

(±std across seeds is small: ≤0.017 for every cell; see
`results/probe1_dyck_depth_summary.json`.)

**Reading 1b.**
- **Every arm learns the training-length task** (≥0.997 @ T=128). Nobody is near
  the 0.182 baseline at any length — there is no *collapse*.
- **The additive counters extrapolate best, and the gap WIDENS with length.**
  At T=128 lstm and e88-linear are tied (1.000 vs 0.999); at T=1024 the order is
  **lstm 0.975 > relu_rnn 0.961 > gdn 0.950 > e88-tanh 0.929 ≈ e88-linear 0.924
  > m2rnn 0.731**. The LSTM↔e88-linear gap grows 0.001→0.051 from T=128→T=1024.
  This is the WGY-2018 ordering (additive non-saturating cells are the best
  counters) and it is seed-stable.
- **But it is a separation of *degree*, not of *kind*.** The linear arms degrade
  *gracefully* (to ~0.92–0.95), they do not fail. **e88-tanh ≈ e88-linear**, so
  *saturation is not the discriminator here* — i.e. this is NOT a clean
  "tanh-cannot-count" demonstration at this cap/drift. The reason is structural
  (see §4): with a display cap of 15 and a near-floor stationary depth
  distribution, the *displayed* target is effectively a **bounded (finite-state)
  counter**, and a running count is itself **linear (`cumsum`)** — the
  comparison (the `max(·,0)` floor / zero-test) is supplied by the **nonlinear
  readout head**, which a linear-*state* recurrence also has. m2rnn is the
  outlier that extrapolates *worst* (0.731), so "more elaborate state" did not
  help counting here.

### 3.2 Variant 1a — `anbncn_viability` (unbounded a^n b^n c^n viability)

Accuracy at each eval length, mean±std over seeds {42,123,456}. Train T=128.
Random baseline = 0.500 (binary). Raw JSONs:
`results/probe1_anbncn_viability__<arm>__seed<seed>.json`.

| arm         | kind                                 | params | T=128 | T=256 | T=512 | T=1024 |
|-------------|--------------------------------------|--------|-------|-------|-------|--------|
| e88-linear  | LINEAR (eigenvalue-rich)             | 7.92M  | 0.999 | 0.960 | 0.868 | **0.812** |
| e88-tanh    | nonlinear-SATURATING (tanh)          | 7.92M  | 0.987 | 0.928 | 0.869 | **0.836** |
| gdn         | LINEAR (gated delta-net)             | 8.25M  | 0.985 | 0.927 | 0.860 | **0.816** |
| m2rnn       | matrix-memory RNN                    | 8.05M  | 1.000 | 0.934 | 0.820 | 0.723 |
| relu_rnn    | ADDITIVE counter (ReLU-Elman) [WGY+] | 7.94M  | 0.763 | 0.757 | 0.746 | 0.731 |
| **lstm**    | **ADDITIVE counter (LSTM)** [WGY+]   | 8.05M  | 1.000 | 1.000 | 0.978 | **0.951** |

**Reading 1a.**
- **The LSTM additive counter extrapolates essentially flat** — 1.000 / 1.000 /
  0.978 / 0.951 across T=128→1024 (std ≤ 0.030) — the signature of having
  learned the *algorithm* rather than the training-length distribution.
- **Every linear / saturating / matrix-state arm degrades steeply:** at T=1024,
  e88-linear 0.812, gdn 0.816, e88-tanh 0.836, m2rnn 0.723 — a 0.16–0.28 drop
  from their (near-perfect) T=128 fit. They *fit* the training length and then
  *fall off* with length. m2rnn (the most elaborate state) extrapolates *worst*.
- **The LSTM↔linear gap widens monotonically with length:** vs e88-linear it is
  0.001 / 0.040 / 0.110 / **0.139** at T=128/256/512/1024 — exactly the
  length-extrapolation separation a real counter predicts.
- **`relu_rnn` is an optimization caveat, not a capability one.** Per seed it is
  bimodal: seed 42 trained and extrapolates *flat like the LSTM*
  (0.969→0.917 @ T=1024 — the counter signature), while seeds 123/456 got stuck
  near 0.64 (the well-known ReLU-RNN dead/exploding-unit fragility). So a bare
  additive cell **does** count when optimization succeeds; the **LSTM is the
  robust positive control**, and it is the one the verdict rests on.

## 4. Verdict

**YES — a genuine additive-counter separation exists in our setup, and it is
cleanest on the unbounded `a^n b^n c^n` task (1a).** The LSTM additive counter
**succeeds at length extrapolation (0.951 @ T=1024, near-flat) where the linear
arms (`e88-linear` 0.812, `gdn` 0.816) AND the saturating `tanh` arm (`e88-tanh`
0.836) all fall off** to 0.81–0.84, and `m2rnn` to 0.72. The gap is seed-stable
and **widens monotonically with length** — the defining footprint of a model
that learned the counter versus models that approximated it at the training
length. The bare ReLU-Elman reproduces the LSTM's flat profile on the seed where
it trained, confirming the mechanism is the **additive / non-saturating cell**
(WGY-2018), not LSTM-specific gating.

**Three honest qualifications** (the result is real but must be stated
precisely):

1. **It is a separation of *degree*, not of *kind*.** The linear arms do not
   collapse to the 0.50 baseline — they *partially* extrapolate (≈0.81). The
   counting comparison is a sign/zero-test that a model can in part supply
   through its **nonlinear readout head** on top of a linear `cumsum` state, so a
   linear-*state* recurrence gets part-way; what it cannot do is keep the count
   *exact* at lengths far beyond training, which is where the additive cell pulls
   ahead.
2. **Saturation is not the clean discriminator here.** `e88-tanh ≈ e88-linear ≈
   gdn` at long T; the separation is "additive LSTM counter beats *all* of them,"
   not "tanh uniquely fails." (This matters: do not frame it as a pure
   tanh-cannot-count demo.)
3. **The capped-Dyck variant (1b) is a much weaker separator** (LSTM 0.975 vs
   e88-linear 0.924 @ T=1024): a display cap + near-floor stationary
   distribution makes the *displayed* target effectively a bounded
   (finite-state) counter, which the linear arms handle. Use **1a** (uncapped,
   count-to-T/3) as the headline; 1b is the cautionary control showing that
   *bounding the counter erases the separation*.

So: **the paper COULD earn a real nonlinearity claim with this task + an additive
(LSTM) counter baseline** — provided it is stated as a *length-extrapolation
(degree) separation on unbounded counting*, with the LSTM (not the bare
ReLU-RNN) as the robust positive control, and acknowledging the linear arms
partially extrapolate.

## 5. Tie-back to S5 and the paper's linear-vs-nonlinear framing

**This is a different axis from the $S_5$ story, and the two are consistent.**

- On **$S_5$** (`review/S5_CONFIG_FLIP.md`) the *linear* `e88-linear` **won**,
  because $S_5$ is a **finite-group word problem → finite-state**, and a
  reflection-rich input-dependent linear recurrence reaches the whole finite-
  state / permutation regime (Grazzi 2025; DeltaProduct 2025). Eigenvalue
  richness is the lever there, and it is a *linear* lever. **No counting is
  involved**, so an additive cell buys nothing — exactly what we see in **1b**,
  where capping the counter makes it finite-state and the linear arms tie.

- On **counting-with-comparison at unbounded length (1a)** the picture **flips**:
  the task is **non-finite-state** (the count must grow with $T$), the linear
  eigenvalue/reflection levers add **no counter magnitude**, and the **additive
  LSTM cell is the only arm that holds up** at T=1024. This is the
  Weiss–Goldberg–Yahav (2018) / Délétang (2022) boundary, orthogonal to the
  Grazzi/DeltaProduct group story.

**Plain implication for the paper's framing.** Our data says the honest
linear-vs-nonlinear claim is **regime-specific, not global**:

> *Finite-state / finite-group tasks (including $S_5$) do **not** separate
> linear-state from nonlinear-state recurrences — eigenvalue-rich linear
> recurrences already reach them (and our `e88-linear` wins $S_5$). The
> separation that **does** hold in our harness is **unbounded counting**: an
> additive (LSTM) counter extrapolates to long sequences where eigenvalue-rich
> linear recurrences (`e88-linear`, GDN) and a saturating-`tanh` state all
> degrade. The nonlinearity that matters is the **additive, non-saturating
> counter cell inside the time-recurrence**, exercised by length extrapolation —
> not the state nonlinearity per se, and not anything a finite-state /
> group word-problem can reveal.*

This both (a) explains why the original $S_5$-as-separator plan failed and
(b) supplies a concrete task + baseline where a real, if graded, nonlinearity
advantage is measurable. A fully *binary* "linear cannot, nonlinear can"
separation would need pushing the linear arms to *collapse* (e.g. exact-match
scoring, longer eval, or a genuinely state-nonlinear `state×state` target —
Probe 2), which this probe does not claim.

---

### Reproduce

```
# tasks + baselines are registered; runner does dynamic idle-GPU scheduling
python experiments/expressivity_tasks/run_probe1_counting.py --tasks dyck_depth        --steps 3000
python experiments/expressivity_tasks/run_probe1_counting.py --tasks anbncn_viability  --steps 3000
python experiments/expressivity_tasks/aggregate_probe1.py --task dyck_depth
python experiments/expressivity_tasks/aggregate_probe1.py --task anbncn_viability
```
Arms, seeds {42,123,456}, train T=128, eval {128,256,512,1024}, fp32,
param-matched ~8M. Raw per-seed JSONs in `experiments/expressivity_tasks/results/`.
