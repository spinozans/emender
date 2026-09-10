# E97 response-gradient qualification: retained v1--v3 attempts

Date: 2026-09-10 UTC. Status: **v3 failed its numerical gate**; a separate
one-knob measurement-only diagnostic completed; `qualification_pass:false` remains.

## Attempt history

- **v1 / `proc_8e00`** stopped in one second, exit 4: the inherited snapshot lacked
  `tests/test_e97_response_boundary_gradient.py`, although the runner named it.
  No tests ran, GPU lease was acquired, or model was loaded.
- **v2 / `proc_0179`** included the missing fixture and passed CPU tests, then
  stopped after 37 seconds, exit 1: the training facade correctly rejected the
  unaligned 17-step kernel input. No numerical comparison or model load occurred.
- **v3 / `proc_f932`** exited 1 after **142 seconds**. It explicitly pads training inputs to multiples of the existing
  sparse checkpoint interval, 16. Added CPU checks verify alignment, preservation
  of real tokens, invalid tail masks, and exactly one supervised opening. Ten
  CPU checks passed in the canonical checkout; frozen collection was also checked
  before launch. Numerical tolerances were not changed.

No failed snapshot was overwritten or automatically retried. Each repaired
attempt has a separate source/recipe identity and retained failure evidence.

Retained v3 root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/response-gradient-qualification-v3`.
Inventory SHA-256:
`21681cc4ca039be8df6f01cca3d1dc5db8f7e4ad6845ea006224f86e03f7fe04`.
Qualifier SHA-256:
`56aa2fbd6bb740848ce33c6951e970e3c4a430bd1f2de2fa70e55a0a46dbc5f4`.
The nominal 17/64-position kernel cases use 32/64 allocated positions and 14/61
valid positions; all extra padding is invalid. The model has 1,166 real tokens,
1,168 aligned prediction-input positions, and 1,169 token slots. Padding changes
no causal prefix or supervised target.

## v3 numerical findings and diagnostic follow-up

The two **linear-state** kernel cases passed: worst input-gradient relative L2
errors were 0.003164 and 0.003117. These do **not** independently qualify the
parent's actual tanh-state branch; the loaded runtime explicitly reports
`linear_state=False`. The follow-up reference assay corrects that coverage gap.

| Full-model case | Opening loss | Worst gradient relative L2 | >5% tensors |
|---|---:|---:|---:|
| Dense reference | 19.50 | 0 | 0 |
| Pilot settings, short prefix | 19.50 | 0.018712 | 0 |
| Active CE73 / MLP256 chunks | 19.25 | 0.060036 | 2 |

All three had zero embedding-output gradient on masked future positions and
nonzero causal-prefix gradients. The active-chunk case was published before its
failed loss-delta assertion. The two tensors above 5% were
`layers.1.mixer.A_log` and `layers.7.mixer.erase_gate_proj.weight`.
The model's final parameter-digest and final authority checks were not reached;
no optimizer step was requested (the existing loader may construct a temporary
Schedule-Free optimizer to restore train/y). This is a **failed qualification**,
not evidence that both modes passed and not yet proof of a backward bug.

The active case changed two chunk controls. To attribute the observed sensitivity,
**`proc_6b80`**, `e97-response-chunk-diagnostic-v1`, measured dense, pilot, CE-only,
MLP-only, combined, and a repeated pilot configuration. It also records head/CE
dtypes, independently scored target NLL, and top-token confidence, and tests the
actual tanh recurrence against Torch autograd. Tolerances remain unchanged;
completion status is explicitly `completed-diagnostic`, distinct from
`qualification_pass`. A completed diagnostic cannot promote the failed v3 gate.

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/response-chunk-diagnostic-v1`.
Inventory: `7f725766c40abb418051754155cdec0554dd9fe7fd28322f5657d482543519ab`.
Qualifier: `1eaff1cd83010cd1f071177d36bcd9042720ea5b45110ececf180f437b41b30c`.

## Completed chunk attribution

`proc_6b80` exited zero in **174 seconds**, with post-run source checks. Its
summary SHA-256 is
`61a151ca11cde655c94d780d72371cf5647f35772d17d281d5abb2365b78ab6b`.
Successful measurement is not successful qualification.

| Case | Reported CE | FP32 NLL from captured logits | Worst gradient relative L2 |
|---|---:|---:|---:|
| Dense | 19.50 | 19.437574 | 0 |
| Pilot | 19.50 | 19.437574 | 0.018712 |
| CE-only change | 19.50 | 19.437574 | 0.018712 |
| MLP-only change | 19.25 | 19.250092 | 0.060036 |
| Combined change | 19.25 | 19.250092 | 0.060036 |
| Pilot repeated | 19.50 | 19.437574 | 0.018712 |

This isolates the observed difference to **MLP chunking** in this bounded assay.
It does not yet distinguish normal BF16 shape-sensitive arithmetic from a defect
in the chunked implementation, and it does not establish why agent acquisition
failed. CE73 versus CE128 did not change these measured gradient metrics.

The **actual tanh-state** recurrence passed independent Torch-autograd comparison:
worst gradient relative L2 errors 0.003171 and 0.003062, below the unchanged 3%
limit. Model parameter digests before/after were identical:
`9250d077297f1755c1393d173acb6c264f9c05093c301d785c99676590289ad2`.
Checkpoint/args/probe hashes also passed before and after.

There is a separate scalar-loss discrepancy: reported CE is FP32, head logits
are BF16 under enabled CUDA autocast, yet CE19.50 differs from the independently
recomputed FP32 NLL19.437574. Top token 12502 has approximately 0.999925 probability
in the pilot case. Neither CE dtype alone nor near-one confidence explains the
observed gap without further checking.

A checkpoint-free synthetic PyTorch CE probe is running as **`proc_5a8d`** at
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/ce-precision-probe-v1`, inventory
`3a0004995bd8112a6d77b348fd0d47c912c677c477890128ad8b813901322c83`.
It compares autocast on/off, explicit FP32 inputs, and manual log-softmax at small
and full vocabulary sizes. It is not by itself an attribution of the model gap.

## Original v1 identity (retained, not the current launch)

Original managed process: **`proc_8e00`**. Intended resource policy, retained by
v3: one exclusively leased local GPU, 3,600-second outer timeout, no automatic retry. Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/response-gradient-qualification-v1`.

Source inventory SHA-256:
`ea2e5a5b518e2f0f4262aa98a521fe00b6ff613b194b5e069733f965f5601093`.
Qualifier SHA-256:
`8040ba1a641c67d0482d3d3059bb141ee9b1dc258d3085ea5a2de31b2d6eaa0f`.

## Frozen questions and checks

1. Compare the actual E97 fused sequential facade against independent FP32 Torch
   autograd at **60 heads / state size 64**, BF16 projections, sequence lengths
   17 and 64, an interior reset, and three invalid tail positions. Include both
   split-edit gates, SiLU Q/K/V, normalized K/Q, output gate, and linear state.
   Require output/state and every input gradient relative L2 error at most 3%;
   absolute-only tolerances must not allow missing small gradients to pass.
2. Load the trusted **4,045,972,080-parameter** parent in explicit train/y mode.
   Use the entire 1,102-token prefix from already-consumed diagnostic
   `consumed-agent-00000996`; supervise **only its first Analysis token, 32750**.
   The remaining 63 existing continuation tokens are present but masked.
3. Compare all parameter gradients between dense CE without grouped layer or
   MLP checkpointing, the pilot's group-size-three / MLP-chunk-4096 settings
   with CE chunks of 128, and active MLP chunks of 256 / CE chunks of 73.
   Frozen tolerances: parameter-relative L2 at most 5%, loss delta at most 0.05.
   Existing projection-recomputation settings are retained in all three cases;
   the dense path is not an independent implementation of every model operator.
4. Verify nonzero gradient reaching the causal prefix and exactly zero embedding
   output gradient on masked future positions. The output head and embedding
   share a parameter; identify that parameter by object identity, not an assumed
   `named_parameters()` alias.
5. Hash model parameters before/after, and checkpoint/args/probe authorities before
   and after. No optimizer step, parameter update, or checkpoint export occurs;
   the existing loader may instantiate an optimizer for its train/y basis restore. CPU BF16 gradient copies are comparison evidence, not master weights
   or CPU optimizer arithmetic.

Full-model comparison receipts are published before their numerical assertions,
so a failing case remains inspectable. The source tree is immutable and includes
its CPU tests, recipe, and runner. All copied `ndm` Python files were checked
against the current canonical checkout before launch. CUDA routing is explicit;
NUMA placement and rank-local Triton caches are separate settings.

## Evidence limits

The prefix is a consumed diagnostic, not independent holdout or new admitted
training data. This assay cannot establish acquisition, generalization, effective
Schedule-Free updates, full 64K packing/backward, or distributed loss scaling.
The 4,096-token MLP threshold is not crossed by this prefix; the separate active
256-token case exercises chunking, not full-64K equivalence.

ADR-003 remains the production distributed-training authority. **R16** from the
compute-pool gap matrix supplies the applicable exact-source/evidence intent;
this single-device numerical assay makes no resilient, native, async, or Frontier
conformance claim and modifies no production recovery behavior.
