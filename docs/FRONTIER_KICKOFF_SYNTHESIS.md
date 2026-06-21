# Frontier Kickoff Synthesis

Task: `synthesize-frontier-handoff`
Date: 2026-06-21

## Scope and Sources Read

This synthesis is a planning baseline for Frontier kickoff. It separates
measured evidence from prior interpretation and from claims that still need
fresh fused-kernel / Frontier evidence before they should influence spend.

Primary sources read:

- `docs/HANDOFF.md` - current self-contained handoff, last updated
  2026-06-20.
- `paper/main.typ` - current Typst paper draft source.
- `paper/README.md` - paper source-of-record and build notes.
- `paper/OUTLINE.md` - historical drafting context, not the release draft.
- `docs/GDN2_E97_NOTES.md` - early E97/GDN2 design note; useful for lineage,
  but its "reference first" experiment plan is superseded by the current
  fused-only project rule.
- `docs/E97_RAW_1P3B_LEADERBOARD.md` and
  `docs/REPRODUCE_E97_RAW_1P3B.md` - older E97-family CMA leaderboard and
  reproduction recipe.
- `docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md` and
  `docs/repro/lb_gdn2_mlp_20260612/bf16_fused_assert.txt` - measured
  GDN2-MLP control and fused/bf16 assertion.
- `docs/repro/lb_emender_mix_20260612/REPRODUCTION.md` - measured full-range
  Emender mixture-axis run.
- `docs/FRONTIER_KICKOFF_QUALITY_PASS.md` - upstream task tightening the
  Frontier kickoff deliverables and fused-only recurrence rule.

Documents intentionally not edited because sibling/downstream tasks own them:
`docs/FRONTIER_ROCM_READINESS_INVENTORY.md` and
`docs/FRONTIER_EXECUTION_GRAPH_DRAFT.md`.

## Non-Negotiable Validation Rule

Any recurrence or state-dynamics validation path must use the fused Triton
recurrence with the fused guard / no eager fallback assertion. Eager or
pure-PyTorch recurrence is invalid for prototypes, quick sanity checks,
debug shortcuts, or preliminary signal checks. If a new probe or state summary
is needed, it must be implemented in the fused kernel and matching backward VJP,
or explicitly logged as a future fused-kernel implementation requirement.

Eager or pure-PyTorch recurrence is invalid, including for prototypes or quick checks.

This rule is stated in `docs/HANDOFF.md` section 5 and reinforced by
`docs/FRONTIER_KICKOFF_QUALITY_PASS.md`. Earlier text in
`docs/GDN2_E97_NOTES.md` that suggested implementing a PyTorch/reference path
before writing Triton is historical context only and must not be used as a
Frontier validation plan.

## Evidence-Backed Baseline

### E97 / Emender Cell

Current handoff definition: E97 is the split-edit gated-delta recurrence with
separate key-axis erase/read and value-axis write gates plus a per-step state
nonlinearity. The canonical update and source map are in `docs/HANDOFF.md`
section 1.2, citing `ndm/triton/e97_chunked.py`,
`ndm/models/e88_fused.py`, and `ndm/models/e88_fla_hybrid.py`
(`use_split_edit=True`).

Evidence-backed details:

- Lead nonlinear arm: `emender` / `nonlin` / E97 uses `f = tanh`, is
  delta-correcting, and is the capability-retaining state-nonlinear cell
  (`docs/HANDOFF.md` section 1.2).
- `e97-linear` keeps split edit and delta correction but removes recurrent
  `tanh` (`f = identity`). In the handoff it is the causal ablation for
  nonlinearity-in-time, not a Frontier arm by itself (`docs/HANDOFF.md`
  sections 1.2 and 2.2; `scripts/cmaes_search_v2.py` comments also identify
  it as "split edit with linear state update").
- Current fused-kernel status is CUDA/Triton, bf16-oriented, and already
  sensitive enough that eager results are non-transferable
  (`docs/HANDOFF.md` sections 3.2 and 5).
- The ROCm/MI250X port is not yet validated; it is the hard P1 prerequisite
  before expensive Frontier scale (`docs/HANDOFF.md` sections 3.2 and 3.3).

### E97-MLP / `emender-mlp`

Current handoff definition: `emender-mlp` is the E97 split-edit delta
token-mixer plus the standard bias-free LLaMA-style SwiGLU MLP channel-mixer.
The handoff gives the seed geometry and measured parameters:
dim 1792, 216 heads, state 32, depth 11, MLP ratio 2.2623,
1,286,589,072 parameters (`docs/HANDOFF.md` section 1.3).

Evidence-backed details:

- The fair comparison is MLP-controlled: `emender-mlp` vs `gdn2-mlp`, not a
  naked E97 cell against GDN2+MLP (`docs/HANDOFF.md` section 1.3).
- The handoff's current LM-loss reading is a convergent-loss null:
  `emender-mlp` and `gdn2-mlp` sit in the same held-out BPB band at matched
  compute, so bulk LM loss does not distinguish the architectures
  (`docs/HANDOFF.md` section 2.1; paper anchor `paper/main.typ` around the
  "same held-out bpb band" discussion).
- In the corrected 1.3B matched-control handoff, `emender-mlp` leads on
  search avg loss 5.8606 vs 5.8949 and non-averaged held-out BPB 2.0911 vs
  2.1013, while `gdn2-mlp` leads only on the averaged-weights basis that the
  handoff flags as inferior/artifact-prone (`docs/HANDOFF.md` section 2.1).

### E97-Linear-MLP / Linear-State E97 Ablation

The task names "E97-linear-MLP"; the current repository evidence mostly uses
`e97-linear` as the state-linear E97 ablation rather than a named production
MLP arm. The handoff treats it as a causal isolation arm: same split-edit
code path, same delta correction, identity state map instead of `tanh`
(`docs/HANDOFF.md` sections 1.2 and 2.2).

Evidence-backed details:

- `e97` vs `e97-linear` isolates the per-step state nonlinearity. The handoff
  states that this comparison is byte-identical code on the same fused kernel
  except `tanh` vs `identity` (`docs/HANDOFF.md` section 2.2).
- In the modular-quadratic length-extrapolation result, linear-state arms
  (`e97-linear` and `gdn2`) memorize the train length and collapse at far
  length while nonlinear `e97` remains length-invariant in decisive cells
  (`docs/HANDOFF.md` section 2.2).
- Older CMA leaderboard evidence ranks `e97-linear` below `e97`, `e97-raw`,
  and `gdn2-mlp` under the 15-minute 1.3B search-loss fitness
  (`docs/E97_RAW_1P3B_LEADERBOARD.md`). That is not a Frontier decision by
  itself because the current fair Frontier pair is `emender-mlp` vs
  `gdn2-mlp`, and because the current handoff supersedes older leaderboard
  interpretations.

### GDN / GDN2 / GDN2-MLP

The paper draft uses GDN as the linear recurrent baseline for the older E88
story. The handoff moves to GDN-2/GDN2-MLP as the fair current control:
GDN-2 is the linear real-axis corner of the Emender taxonomy, and
`gdn2-mlp` is the GDN-2 mixer plus the same SwiGLU MLP channel-mixer
(`docs/HANDOFF.md` sections 1.1 and 1.3; `paper/main.typ` taxonomy
proposition).

Evidence-backed details:

- The GDN-2 special-case relation is structural: GDN-2 is the
  `{decay, reflect} x linear` sub-grid of the Emender taxonomy
  (`docs/HANDOFF.md` section 1.1; `paper/main.typ` taxonomy figure and
  proposition).
- Current `gdn2-mlp` geometry in the handoff: dim 2176, 30 heads, depth 12,
  MLP ratio 3.2587, use conv, 1,286,713,448 parameters
  (`docs/HANDOFF.md` section 1.3).
- The fresh measured GDN2-MLP control converged over 104 evals with best
  avg-loss 5.8949, and its live best-eval args assert bf16 and fused
  GDN2 kernel use (`docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md`;
  `docs/repro/lb_gdn2_mlp_20260612/bf16_fused_assert.txt`).

### Fused Triton Kernels and Current Validation

Ready facts:

- Fused recurrence is a hard methodology guardrail, not a performance
  preference (`docs/HANDOFF.md` section 5; `docs/FRONTIER_KICKOFF_QUALITY_PASS.md`).
- CUDA-side fused E97 code and tests exist (`ndm/triton/e97_chunked.py`,
  `ndm/models/e88_fused.py`, `tests/test_e97_chunked.py`, cited in
  `docs/HANDOFF.md` artifact map).
- GDN2-MLP evidence records bf16 + fused official GDN-2 kernel usage with no
  eager T-scan fallback for the GDN-2 family
  (`docs/repro/lb_gdn2_mlp_20260612/bf16_fused_assert.txt`).

Open validation status:

- E97 on Frontier/ROCm is not validated. The handoff calls the bf16-only
  fused E97 kernel port the biggest engineering risk and requires parity at
  T in {128, 512, 1024, 2048} on a single MI250X GCD before multi-node runs
  (`docs/HANDOFF.md` sections 3.2 and 3.3).
- There is no fp32 safety net for the fused E97 path, and CUDA-side history
  already includes chunked overflow / NaN edge cases
  (`docs/HANDOFF.md` section 3.2).

## Paper Draft Claims and Transfer Risk

The current paper source is `paper/main.typ`; `paper/README.md` identifies it
as the master source and `paper/OUTLINE.md` as historical context.

### Claims from E88/GDN That Are Paper-Backed

- A 1.3B-class E88/Emender checkpoint and GDN checkpoint reach the same
  sub-1-BPB language-model band on The Pile; the paper states E88 0.973
  training BPB and GDN 0.973 at released v0.3 checkpoints, and held-out
  BPB tie behavior across slices (`paper/main.typ` loss-vs-wallclock and
  held-out BPB sections).
- The paper's 8M state-tracking probes show Emender ahead of GDN on S5 at
  training length and under length extrapolation: Emender 0.7918 vs GDN
  0.3552 at S5 T=128, with both declining at longer T but Emender falling
  slower (`paper/main.typ` expressivity section).
- The paper's 1.3B fine-tune probe shows the delta E88 model learns S5 at
  trained length much better than linear GDN, while all models degrade with
  length. Under the symmetric 24,000-step S5 budget, E88 is 0.921 at T=64
  while GDN is 0.117; at T=1024 E88 is 0.076 and GDN 0.015
  (`paper/main.typ` "same separation at 1.3 B scale").
- The paper is careful that E88 does not solve arbitrary-length S5 at fixed
  width; it claims a budget-robust ordering and trained-length competence,
  not unlimited length generalization (`paper/main.typ` S5 discussion).

### Claims That May Transfer to E97/GDN2

- Bulk LM loss as a poor distinguisher likely transfers in spirit because
  the latest handoff reproduces a convergent-loss-null reading in the
  current `emender-mlp` vs `gdn2-mlp` pair (`docs/HANDOFF.md` section 2.1).
- The state-nonlinearity mechanism transfers more directly where the arm is
  actually E97 vs `e97-linear` on the same fused kernel; this is stronger
  current evidence than the older E88/GDN paper framing for Frontier
  architecture selection (`docs/HANDOFF.md` section 2.2).
- The GDN/GDN2 relation is a taxonomy bridge, not a performance proof:
  GDN-2 is the linear real-axis corner of the Emender grid, but old GDN
  numbers should not be treated as measured GDN2-MLP Frontier numbers
  without the current `gdn2-mlp` artifacts (`paper/main.typ`;
  `docs/HANDOFF.md`; `docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md`).

### Claims That Need Fresh Frontier Evidence

- "The Triton kernel runs identically on CUDA and ROCm" appears in
  historical paper-outline language (`paper/OUTLINE.md`), but current
  handoff evidence says the E97 ROCm/MI250X port is still a hard open risk.
  For planning, use the handoff status, not the outline wording.
- Any statement that E97/GDN2 will retain the same qualitative capability
  divergence after long-tail commapile scaleout is a hypothesis. The handoff
  explicitly frames the central open question as whether the proved
  small-scale modular-quadratic separation surfaces as measurable real-LM
  capability near the emergence regime (`docs/HANDOFF.md` section 6).
- DiLoCo parity is only measured locally through I<=4, with seed/outer
  optimizer observations through I<=6. The 512-GCD / large-island Frontier
  regime is explicitly untested (`docs/HANDOFF.md` sections 3.1 and 6).

## Prior Interpretation / No-Go Framing Not Accepted as Fact

The handoff warns that older auto-verdicts and no-go statements were sometimes
based on confounded comparisons, averaged-weight artifacts, under-search,
non-symmetric geometry, grok-suppressed separator batteries, or inherited
interpretation (`docs/HANDOFF.md` sections 2.1 and 5).

Do not carry forward these interpretations as conclusions:

- The old `lb_compare` auto-verdict "clean NO-GO / gdn2-mlp best all-around"
  is superseded by the corrections in the current handoff. It is not a
  Frontier conclusion unless re-established with fresh fused-kernel evidence
  and a cleared confound audit (`docs/HANDOFF.md` section 2.1).
- Earlier E97 throughput-speedup claims from grok-scale microbenchmarks are
  refuted at 1.3B matched conditions. The current evidence is a per-token
  throughput tie at matched bs4, with an Emender memory/batch-size advantage
  rather than a raw kernel-speed advantage (`docs/HANDOFF.md` section 2.3).
- Older "width closes the gap" / capacity interpretations are superseded by
  length-extrapolation controls in the modular-quadratic result
  (`docs/HANDOFF.md` section 2.2).
- M1/M2 architecture extensions are defended nulls in the handoff and should
  not be added to the Frontier arm list without new scoped evidence
  (`docs/HANDOFF.md` section 2.4).

Policy for this kickoff: prior no-go statements are not accepted as conclusions without fresh fused-kernel evidence, Frontier/ROCm evidence where the claim concerns Frontier, and an explicit confound audit.

## Open Questions for Frontier Kickoff

### ROCm / HIP / MI250X

- Can the bf16-only fused E97 split-edit kernel be ported to ROCm/MI250X with
  acceptable parity against CUDA fused behavior at T in {128, 512, 1024, 2048}
  and no eager fallback?
- What is the single-MI250X-GCD E97 throughput and memory ceiling for the
  target `emender-mlp` geometry? This measurement should replace planning
  estimates in token-per-24h calculations.
- Does GDN2-MLP run through a mature ROCm-compatible FLA path on this stack, or
  does it also need local port work? It is the fallback/control arm, so this
  needs debug-queue evidence rather than assumption.
- Are bf16, Triton/HIP compilation, torch/ROCm versions, and any `torch.compile`
  choices stable enough to avoid the known ROCm bf16 NaN gotchas flagged in
  `docs/HANDOFF.md` section 3.2?

### Debug Queue Testing

- First smoke: single-GCD fused E97 forward/backward parity and training step
  with fused guard / no eager fallback. A run without the guard is invalid.
- Second smoke: single-GCD `gdn2-mlp` fused path, confirming the actual ROCm
  operator path and no silent fallback.
- Third smoke: tiny end-to-end train/resume/checkpoint/eval loop on commapile,
  because Frontier releasable models must be commapile from seed onward.
- Fourth smoke: small multi-node RCCL path with the exact communication stack
  planned for scale. The handoff specifically calls out ROCm/Megatron-LM,
  `aws-ofi-rccl` on Slingshot, pre-built DeepSpeed JIT ops if used, and
  PyTorch DCP sharded checkpointing (`docs/HANDOFF.md` section 3.2).

### Extended 64 x 24h Runs

- The handoff records 64 nodes x 24 h as the `extended`-partition planning
  shape and estimates 150-180B tokens per 24h at 64 nodes, but flags scheduler
  and throughput numbers as external/planning estimates to re-verify
  (`docs/HANDOFF.md` sections 3.2 and 7).
- The execution plan should pack E97 and GDN2 into one 64-node job when both
  are being compared, so the two arms stay concurrent under the one-running
  job constraint (`docs/HANDOFF.md` section 3.2).
- Before committing extended runs, require debug-queue proof that checkpoint
  resume, data streaming, fused guards, held-out eval, and job dependency
  chaining all work on Frontier.

### Horizontal Single-GPU-Island Scaleout

- Local evidence says no-NVLink DDP wastes about 48% of GPUs, while independent
  single-GPU processes scale near-linearly; that is the reason DiLoCo is the
  local parallelism path (`docs/HANDOFF.md` section 3.1).
- Frontier planning should distinguish horizontal single-GCD islands from
  synchronous DDP. The large-island/512-GCD setting is not validated by the
  I<=4/I<=6 local evidence.
- Open question: what island granularity is actually best on MI250X after
  measuring fused E97 and GDN2 throughput? One GCD per island minimizes inner
  communication; node-level islands may improve local efficiency but require
  validated RCCL behavior.

### DiLoCo

- Ready local recipe: plain averaging, `--diloco_k 250`,
  `--diloco_outer_lr 1.0`, `--diloco_outer_beta 0.0`; outer momentum is unsafe
  in measured local runs (`docs/HANDOFF.md` section 3.1).
- The Schedule-Free merge bug is fixed locally by averaging both x and z and
  preserving the Schedule-Free clock scalars; if the long-horizon run uses
  AdamW+cosine rather than Schedule-Free, this interaction may be moot but the
  optimizer choice must be fixed before P2 (`docs/HANDOFF.md` sections 3.1
  and 3.2).
- DiLoCo matched-token parity is measured through I<=4 locally and should not
  be extrapolated to 512 GCDs as fact (`docs/HANDOFF.md` sections 3.1 and 6).
- Multi-node DiLoCo over RCCL plus DP/3D inner training is an unpublished
  composition in this project and requires its own throughput, parity, and
  fault-tolerance validation (`docs/HANDOFF.md` section 3.2).

## Coordination Notes for ROCm Inventory and Execution Planning

Ready to hand to ROCm readiness inventory:

- E97 fused source targets: `ndm/triton/e97_chunked.py`,
  `ndm/models/e88_fused.py`, and tests `tests/test_e97_chunked.py` as cited
  by the handoff.
- E97 validation requirement: bf16 fused parity on MI250X GCD at T in
  {128, 512, 1024, 2048}, no fp32 safety net, no eager fallback
  (`docs/HANDOFF.md` section 3.2).
- GDN2-MLP control evidence: fresh 104-eval control and fused/bf16 assertion
  in `docs/repro/lb_gdn2_mlp_20260612/`.
- Known systems risks to inventory: Triton/HIP compile behavior, bf16 NaNs,
  RCCL over Slingshot with `aws-ofi-rccl`, checkpoint format, and data staging.

Ready to hand to execution planning:

- Fair arm pair: `emender-mlp` vs `gdn2-mlp`; do not add M1/M2 arms to the
  Frontier scaleout without new evidence (`docs/HANDOFF.md` sections 1.3 and
  2.4).
- Legal data rule: Frontier/releasable path must be commapile end-to-end;
  `pile.txt` is local gate only (`docs/HANDOFF.md` section 3.2).
- Local/Frontier gate order: P0 local equal-FLOPs race, P1 ROCm kernel port,
  P2 small-multinode DiLoCo/RCCL validation, P3 scale
  (`docs/HANDOFF.md` section 3.3).
- DiLoCo recipe default: plain average with outer beta 0.0; treat any outer
  momentum or large-island extension as a new experiment, not an inherited
  default (`docs/HANDOFF.md` section 3.1).

Requires fresh evidence before influencing spend:

- E97 ROCm viability and single-GCD throughput.
- GDN2-MLP ROCm operator path and throughput.
- 64-node x 24h token-throughput estimate.
- 512-GCD DiLoCo learning parity and fault tolerance.
- Any claim that the small-scale modular-quadratic separation will surface as
  real-LM capability after long-tail commapile scaleout.
- Any inherited no-go verdict about E97/Emender unless it is reproduced under
  fused-kernel, fair-geometry, MLP-controlled, Frontier-relevant conditions.

## Kickoff Bottom Line

Evidence-backed: the current fair architecture pair is `emender-mlp` vs
`gdn2-mlp`; bulk LM loss is a convergent null rather than a reliable selector;
E97's per-step nonlinearity has a stronger current mechanism result against
linear-state ablations; 1.3B throughput is a matched per-token tie with a memory
advantage for Emender; and local DiLoCo's safe recipe is plain averaging.

Not yet evidence-backed for Frontier: the ROCm/HIP E97 fused kernel, the exact
MI250X throughput, 512-GCD DiLoCo learning parity, and the emergence of the
small-scale expressivity separation as real-LM capability at long-tail scale.
Those are the debug-queue and extended-run agenda, not settled facts.
