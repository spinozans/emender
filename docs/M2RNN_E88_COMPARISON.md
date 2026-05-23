# M2RNN and E88 comparison plan

## Positioning

M2RNN is useful evidence for the same theoretical direction as E88: nonlinear
recurrent updates over matrix-valued state are a natural answer to state-capacity
limits in linear recurrent models. The clean framing is not that matrix state is
unique to E88. It is that M2RNN validates the premise, while E88 tests a deeper
claim: a pure nonlinear recurrent stack can be trained efficiently at 1.27B
parameters and can match the best linear recurrent baselines on language-model
quality.

The core empirical distinction to test is therefore:

- M2RNN: nonlinear matrix-state recurrence used primarily as a hybrid component.
- E88: pure nonlinear recurrent LM, optimized end to end with the Triton kernel.
- Expressivity axis: state-tracking tasks that are motivated by the theory but
  are not the central evaluation in M2RNN.

Avoid broad priority claims like "first nonlinear matrix-state RNN." The stronger
claim is narrower and more defensible: pure E88 can train at production scale,
does not need attention or linear-recurrent layers to work, and hybrids can
degrade the state-tracking behavior that nonlinear recurrence is meant to supply.

## Implemented harness support

Commit `5917a67` adds M2RNN support to the expressivity harness:

- `m2rnn`: tied-head M2RNN geometry used by the local search path.
- `m2rnn-paper`: grouped-head geometry matching the released M2RNN configs
  more closely: one q/k head, many v/f/g/W heads, K=64, V=`n_state`.
- Hybrid patterns can now include `m2rnn` and `m2rnn-paper` alongside `E88` and
  `fla-gdn`.

The harness disables bf16 autocast for M2RNN-containing expressivity runs because
the upstream XMA Triton path currently has a bf16-autocast compile edge case in
small synthetic-task shapes. This does not change the production Pile trainer.

## Current focused sweep

The first comparison should be small and decisive: use the existing canonical
paper protocol and run the state-tracking tasks M2RNN did not emphasize.

Command launched:

```bash
XMA_PATH=/tmp/m2rnn_xma PYTHONPATH=/tmp/m2rnn_xma \
python3 experiments/expressivity_tasks/run_canonical_sweep.py \
  --gpus 0 4 5 6 \
  --tasks modular_counter fsm_tracking parity \
  --patterns pure_M2RNN pure_M2RNN_paper \
             hybrid_GDN_M2RNN_single hybrid_GDN_E88_single \
  --seeds 42 123 456 \
  --output_dir /tmp/m2rnn_expressivity_canon
```

Protocol:

- dim 384
- depth 4
- n_heads 32
- n_state 32
- schedule-free AdamW
- 10K steps
- batch size 32

Output is intentionally under `/tmp` until the results are summarized:

```text
/tmp/m2rnn_expressivity_canon
```

## M2RNN-specific expressivity target

The comparison should now focus on M2RNN, not GDN. GDN is useful background
because it shares a delta-memory lineage with E88, but M2RNN is the direct
nonlinear matrix-state competitor.

The new `keyed_fsm_memory` task is designed for this comparison:

- each sequence maintains a table of keyed finite-state values;
- operations are encoded as `KEY OP`;
- some operations are absolute writes, while others transform the old value at
  that key modulo the state count;
- the final query asks for the current state of a randomly selected key.

This hits the E88/M2RNN difference directly. E88 has a delta-correcting,
key-dependent transition. M2RNN has nonlinear matrix state, but the paper-form
candidate is a raw outer-product write through a fixed right transition:
`tanh(H W + k v^T)`.

Run the focused M2RNN comparison with:

```bash
python3 experiments/expressivity_tasks/run_canonical_sweep.py \
  --gpus 0 5 6 \
  --tasks keyed_fsm_memory overwrite_recall reset_recall \
          modular_counter fsm_tracking parity \
  --patterns pure_E88 pure_M2RNN pure_M2RNN_paper \
  --seeds 42 123 456 \
  --output_dir /tmp/m2rnn_e88_expressivity
```

The main readout is not only final accuracy. Track time-to-threshold and length
extrapolation:

- `overwrite_recall` / `reset_recall`: delta-correcting write advantage;
- `modular_counter` / `fsm_tracking` / `parity`: recurrent nonlinear
  state-tracking advantage;
- `keyed_fsm_memory`: combined keyed overwrite plus state-dependent update.

## Readout

Compare against the existing canonical E88/FLA/hybrid table in
`experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`.

Important comparisons:

- `pure_M2RNN_paper` vs `pure_E88`
- `hybrid_GDN_M2RNN_single` vs `pure_M2RNN_paper`
- `hybrid_GDN_E88_single` vs `pure_E88`
- both hybrids vs `pure_FLA`
- E93 variants vs `pure_E88`, as a minimal single-matrix-state nonlinear
  recurrent control without the E88 head structure.

If M2RNN-paper is weak on modular counter or FSM tracking, that supports the
claim that the paper validated the matrix-state premise but did not optimize the
expressivity-relevant pure recurrent regime. If hybrids underperform pure E88,
that reinforces the existing result that hybridization is not the right default
for the computational-expressivity axis.

## E93 follow-up

E93 is the right minimal-control axis: one rectangular matrix state per layer,
delta update, learned row transform, and no explicit head abstraction. Recent
E93 ablations add optional output gating and bounded residual output, so it can
separate "matrix-state nonlinear recurrence is enough" from "E88's shaped
many-head design is necessary."

Useful quick sweep:

```bash
python3 experiments/expressivity_tasks/run_suite.py \
  --tasks modular_counter fsm_tracking parity \
  --models E93 E93_no_decay \
  --seeds 42 123 456 \
  --dim 384 --depth 4 \
  --output_dir /tmp/e93_expressivity_controls
```

## Live M2RNN stability report, 2026-05-09

Reference paper: Mishra, Tan, Stoica, Gonzalez, Dao, "M2RNN:
Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling,"
arXiv:2603.14360. The paper validates the matrix-state nonlinear-RNN direction,
but the current local evidence says the published head geometry is not the
stable object in this codebase.

### Production 1.27B observations

All runs below use 2K context, byte/Pile training, bf16, global grad clipping 1,
and schedule-free AdamW unless otherwise noted.

| model | shape | status | latest readout |
| --- | --- | --- | --- |
| E88 | dim 1664, depth 12, H=370, N=32, bs=5 | stable | loss near 3.0 after long convergence, ~7.7K tok/s |
| M2RNN tied/CMA-ES | dim 1920, depth 21, H=370, N=16, bs=5 | stable | step 9250 loss 4.085, grad 1.48, ~7.5K tok/s |
| M2RNN paper-shaped | dim 3072, depth 10, H=759, N=16, bs=4 | stopped | step 8400 loss 11.574, grad 4.19e7, ~5.4K tok/s |
| M2RNN paper LR 2e-4 | same paper shape | stopped | step 1600 loss 11.004, grad 2.74e7 |
| M2RNN paper LR 1e-4 | same paper shape | stopped | step 1600 loss 14.696, grad 1.73e7 |
| GDN + paper M2RNN | 3 GDN : 1 paper-M2 pattern | stopped | step 1100 loss 45.124, grad 52.75 |

The important contrast is not "M2RNN fails." The tied/CMA-ES M2RNN trains
smoothly. The specific failure is the paper-shaped grouped-head geometry under a
pure schedule-free LM run.

### Working hypothesis

The instability appears to be gradient conditioning, not unbounded recurrent
state. In the paper-shaped configuration, the hidden state is bounded by tanh
and forget interpolation, but parameter gradients still become huge.

The likely mechanism is:

- Paper shape uses one shared query/key address stream by default, repeated
  across hundreds of value/forget/gate/weight heads. Gradients from all those
  heads collapse back through a very small address path.
- E88 and the tied/CMA-ES M2RNN use many independent address programs, which is
  closer to the local "multiprogramming" picture.
- E88 also L2-normalizes keys/queries and uses a delta write
  `v - S @ k`, so write scale and error correction are controlled at every
  step.
- Paper-shaped M2RNN writes a raw outer-product candidate through
  `tanh(h W + k v^T)`. Forward values are bounded, but the projection/gate
  gradients can still be badly conditioned.

This is reportable as optimizer/geometry robustness: with the same local
training style, pure E88 is smooth, tied M2RNN is smooth, and paper-shaped M2RNN
is brittle.

### New ablation knobs

Current code adds an explicit M2RNN query/key stabilizer without changing the
paper-shaped default:

```bash
--m2rnn_q_heads <int>
--m2rnn_k_heads <int>
--m2rnn_normalize_qk 1
--m2rnn_use_residual 0
--m2rnn_freeze_state_weight 1
```

These are threaded through:

- `train.py`
- `elman/models/m2rnn_baseline.py`
- `elman/models/hybrid_ladder.py`
- `experiments/expressivity_tasks/train_task.py`
- `experiments/expressivity_tasks/train_hybrid.py`

The q/k-sharing production probe is:

```bash
GPUS="0 5 6 7" QK_HEADS="3 11 23 69" NORMALIZE_QK=0 \
STEPS=1500 LR=1e-4 LAUNCH_DELAY=75 \
./run_m2rnn_qk_ablation.sh
```

Because H=759 in the paper-shaped run, q/k head counts must divide 759. The
first sweep avoids qk=253 and qk=759 at fixed dim because those add very large
q/k projection parameter counts and are no longer close to the paper-shaped
1.27B object.

First attempt: qk=3 at bs=4 loaded a 1.312B parameter model but OOMed on the
first 2K-context step. The larger q/k variants need a memory-reduced probe.
The current active probe therefore uses microbatch 1 with grad accumulation 4:

```bash
GPUS="0 5 6" QK_HEADS="1 3 11" NORMALIZE_QK=0 \
STEPS=1000 LR=1e-4 BATCH_SIZE=1 GRAD_ACCUM=4 LAUNCH_DELAY=75 \
./run_m2rnn_qk_ablation.sh

GPUS="7" QK_HEADS="1" NORMALIZE_QK=1 \
STEPS=1000 LR=1e-4 BATCH_SIZE=1 GRAD_ACCUM=4 \
./run_m2rnn_qk_ablation.sh
```

Early readout:

| variant | params | step 50 | step 100 |
| --- | ---: | --- | --- |
| qk=1, raw | 1.304B | loss 40.772, grad 10.62 | loss 31.849, grad 5.19 |
| qk=3, raw | 1.312B | loss 40.876, grad 5.81 | loss 31.873, grad 5.62 |
| qk=11, raw | 1.343B | loss 40.886, grad 5.62 | loss 31.885, grad 5.50 |
| qk=1, normalized | 1.304B | loss 43.772, grad 38.00 | loss 33.631, grad 19.00 |

Step 150-500 continues the same pattern: raw qk=1, qk=3, and qk=11 remain on
nearly the same loss curve. At step 500, qk=1 is loss 27.224 and qk=3 is loss
27.338. Normalized qk=1 starts worse and only catches up toward the same curve.
This argues against "shared q/k is the whole problem." q/k sharing affects early
gradient scale, but the paper-shaped geometry remains far from the tied-M2RNN
and E88 curves even after reducing microbatch memory. The next likely culprit is
the raw M2RNN update/projection geometry: the `tanh(h W + k v^T)` candidate,
trainable `state_weight`, large D*v residual path, or the output projection
fan-in.

The next ablation wave should keep qk=1, bs=1, grad accumulation 4, and compare:

| variant | purpose |
| --- | --- |
| `--m2rnn_freeze_state_weight 1` | test whether learned `W` conditioning is the source |
| `--m2rnn_use_residual 0` | test whether the direct D*v path dominates early loss/gradients |
| both together | test a more minimal bounded M2RNN recurrence |

### Decision rule

If increasing q/k heads reduces pre-clip grad norms and brings loss down toward
the tied-M2RNN curve, then the paper result is missing the same practical
ingredient E88 found: many independent normalized address programs. If
`--m2rnn_normalize_qk` fixes the qk=1 case, then normalization is the main
stabilizer. If neither helps, the issue is more likely the raw M2RNN update or
trainable state-weight conditioning, and the next ablation is freezing or
spectrally constraining `state_weight`.
