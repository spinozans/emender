# Frontier orientation brief - Emender/E97/GDN2

Task: `frontier-orient-handoff-paper`

Date: 2026-06-21

## Source scope

This brief reads the current handoff, `docs/HANDOFF.md`, and the current paper draft, `paper/main.typ`. It is neutral by construction: it separates directly observed repository evidence, paper claims, prior-agent or handoff interpretations, hypotheses to test, and open questions. Go/no-go wording below is a recommendation or hypothesis unless tied to cited evidence.

Primary citations:

- Handoff orientation and architecture: `docs/HANDOFF.md` lines 16-37, 53-171.
- Handoff evidence and Frontier plan: `docs/HANDOFF.md` lines 183-333, 337-620, 698-727, 739-764.
- Paper claims and scope limits: `paper/main.typ` lines 320-459, 600-663, 1270-1428, 2040-2088, 2570-2625, 2660-2677.
- Prior-agent correction/disagreement artifact: `experiments/lb_compare_20260613/LEADERBOARD.md` lines 1-24 and 60-111; `experiments/lb_compare_20260613/REPRODUCTION.md` lines 51-69.
- Frontier batch quality pass: `docs/FRONTIER_ROCM_SCALEOUT_QUALITY_PASS_20260621.md` lines 30-41 and 46-63.

## Neutral working summary

The program is comparing recurrent token-mixer architectures whose language-model loss appears to converge into a narrow band, while their claimed differences are expected to show up on state-tracking capability and long-horizon behavior rather than short-run perplexity alone.

The active Frontier-relevant comparison is `emender-mlp` versus `gdn2-mlp`, both approximately 1.286B parameters and both with a SwiGLU MLP channel mixer. In the handoff, `emender-mlp` is described as E97 split-edit delta recurrence plus MLP, while `gdn2-mlp` is GDN-2 mixer plus the same MLP (`docs/HANDOFF.md` lines 141-170). The paper's older E88 framing is related but not identical to the current Frontier arm naming: E88 is the 1.3B Emender instance in the paper (`paper/main.typ` lines 246-247, 791-797), while the handoff says E97/E88/E98/E99 should now be treated as historical run identifiers and dynamics names should be preferred (`docs/HANDOFF.md` lines 98-101).

The immediate expensive-run gate should not be "E97 is proven better" or "GDN2 is the no-go default." The gate should be whether debug-queue evidence shows that the Frontier software stack can run both arms correctly, measure comparable throughput/loss, and produce enough small-scale capability and convergence evidence to justify a longer allocation. The upstream quality pass already made that gating explicit: debug evidence and allocation accounting come before extended-queue packages (`docs/FRONTIER_ROCM_SCALEOUT_QUALITY_PASS_20260621.md` lines 34-41, 46-63).

## Observed evidence in the repo

### Architecture and model construction

- `emender-mlp` is documented as E97 split-edit delta plus a bias-free LLaMA-style SwiGLU MLP, with dim 1792, 216 heads, state 32, depth 11, MLP ratio 2.2623, and 1,286,589,072 measured parameters (`docs/HANDOFF.md` lines 141-158).
- `gdn2-mlp` is documented as GDN-2 mixer plus the same SwiGLU MLP, with dim 2176, 30 heads, depth 12, MLP ratio 3.2587, and 1,286,713,448 measured parameters (`docs/HANDOFF.md` lines 160-170).
- `e97-linear` is the same split-edit delta construction with the per-step state nonlinearity removed: `f = identity` instead of `tanh` (`docs/HANDOFF.md` lines 124-137). In code generation it is a named CMA search model type and emits `--linear_state 1` (`scripts/cmaes_search_v2.py` lines 183-187, 916-931). The brief treats `e97-linear-MLP` as the MLP-wrapped version of this linear-state ablation that debug experiments should run if available under the same model wrapper; if the exact CLI arm name differs, the debug task should record that.
- GDN/GDN2 are treated in the handoff and paper as linear-state gated-delta baselines. The paper says GDN-2 is the real-diameter, no-saturation sub-grid of the Emender family (`paper/main.typ` lines 650-663), and the handoff repeats that claim (`docs/HANDOFF.md` lines 89-96).

### Loss and perplexity convergence

- The handoff reports `emender-mlp` and `gdn2-mlp` within the same held-out BPB band at a 15-minute matched budget: search avg-loss 5.8606 versus 5.8949, non-averaged held-out BPB 2.0911 versus 2.1013, and averaged held-out BPB 2.1783 versus 2.1550 (`docs/HANDOFF.md` lines 183-221; `experiments/lb_compare_20260613/LEADERBOARD.md` lines 14-24).
- The paper reports E88, GDN, and M2RNN-CMA in a sub-1-BPB held-out/statistical tie on The Pile at released v0.3 checkpoints, and says bulk held-out perplexity does not distinguish the architectures at that budget (`paper/main.typ` lines 1362-1428).
- The paper explicitly limits this evidence: the 1.3B runs are single-seed per architecture, and whether the same band holds at larger parameter counts, larger corpora, or longer training remains open (`paper/main.typ` lines 1417-1428, 2660-2677).

### Capability and nonlinearity-in-time

- The handoff reports a capability separation on `modular_quadratic`: nonlinear `e97` remains length-invariant in a cited p=256/dim1024 example, while `e97-linear` and `gdn2` collapse by T=4096; the handoff frames this as isolating per-step nonlinearity-in-time because the e97 versus e97-linear comparison differs only by `tanh` versus identity (`docs/HANDOFF.md` lines 223-247).
- The handoff also reports predicted negative controls: no separation on a contractive `iterated_nonlinear_map`, and no stable e97-over-gdn2 result on additive `anbncn_viability` (`docs/HANDOFF.md` lines 254-273). This matters because it argues against the overclaim "nonlinear-in-time always beats linear."
- The paper's formal section names a gap: the trusted core proves k-step separation on a constructed alphabet, but not the stronger S5 generator-specific capacity bound; empirical sections currently stand in for that stronger claim (`paper/main.typ` lines 2040-2088).

### Throughput, DiLoCo, and long-horizon recipe

- The handoff reports a 1.3B per-token throughput tie on RTX 6000 Ada: emender/gdn2 matched-conditions ratio 0.976x, with Emender using less memory at matched batch and fitting a larger batch (`docs/HANDOFF.md` lines 285-305).
- Local DDP scaling is reported as poor on the no-NVLink box, motivating DiLoCo: 7-GPU DDP at 31,291 tok/s and about 52% efficiency, while independent processes approach 62,000 tok/s (`docs/HANDOFF.md` lines 367-379).
- The local DiLoCo recipe supported by repo artifacts is plain averaging, `--diloco_outer_beta 0.0`; outer momentum is reported to diverge (`docs/HANDOFF.md` lines 381-394).
- The handoff says the Schedule-Free merge bug was fixed by averaging both `x` and `z` while preserving clock scalars, and cites `train.py:diloco_merge`, `tests/test_diloco_merge.py`, and `evaluations/sf_diloco_merge_smoke/README.md` (`docs/HANDOFF.md` lines 396-407).
- The long-horizon training recipe recommendation is AdamW with warmup and cosine decay, not constant-LR schedule-free, because the latter was associated with both DDP and DiLoCo held-out BPB collapse in prior runs (`docs/HANDOFF.md` lines 475-488).

### Frontier-specific evidence and gaps

- Frontier is MI250X/ROCm; the handoff identifies the E97 fused split-edit Triton kernel port as the hard prerequisite and largest engineering risk, requiring bf16 parity at T in {128,512,1024,2048} and single-GCD throughput before multi-node runs (`docs/HANDOFF.md` lines 595-601, 609-612).
- The handoff states that commapile is mandatory for releasable Frontier training and that `pile.txt` is not license-clean for the releasable path (`docs/HANDOFF.md` lines 531-551).
- The 64-node extended-queue throughput estimate and several scheduler details are explicitly flagged as external OLCF/operator information, not internal measured repo evidence, and must be re-verified before allocation decisions (`docs/HANDOFF.md` lines 505-524, 739-764).
- The handoff says DiLoCo evidence is only through I <= 4 locally, seed-maturity evidence only through I <= 6, and the 512-GCD/large-island Frontier regime remains untested (`docs/HANDOFF.md` lines 431-437, 455-461, 712-721).

## Paper claims

- The paper claims pure nonlinear recurrent LMs can train at the 1.3B class without time-axis parallelism, with E88 reaching 0.973 BPB on The Pile after about 23 GPU-days (`paper/main.typ` lines 320-334).
- The paper claims the delta-correcting update is strictly more expressive than raw-write in the formalized matched-signature resource class, with empirical S5 learning-efficiency support (`paper/main.typ` lines 336-357).
- The paper claims E88 lands in the same loss-versus-wallclock band as GDN at the 1.3B class under matched per-architecture CMA-ES (`paper/main.typ` lines 359-363).
- The paper frames bulk LM loss as a measured null result and says loss does not license the train-loss ordering as an architecture verdict (`paper/main.typ` lines 420-425).
- The paper treats GDN-2 as a special case of Emender in the taxonomy, occupying the linear real-axis/no-saturation corner (`paper/main.typ` lines 650-663).
- The paper lists several predictions, including that width-axis multi-programming should scale beyond 1.3B without throughput collapse and that the Emender state-tracking advantage should persist at longer token budgets; these are explicitly falsifiable predictions, not current evidence (`paper/main.typ` lines 2570-2584).

## Prior-agent interpretations and disagreements

- Prior leaderboard text contains a strong "gdn2-mlp is best all-around" conclusion based on the same `lb_compare` artifact (`experiments/lb_compare_20260613/LEADERBOARD.md` lines 60-111). That conclusion is explicitly preceded by corrections saying the verdict overstates against Emender, that `emender-mlp` is E97 delta plus MLP, and that the primary metrics lean Emender within noise (`experiments/lb_compare_20260613/LEADERBOARD.md` lines 1-8).
- `experiments/lb_compare_20260613/REPRODUCTION.md` likewise marks its original verdict as superseded by corrections and says the separator battery was grok-suppressed and not a decisive test of the nonlinear-in-time claim (`experiments/lb_compare_20260613/REPRODUCTION.md` lines 51-69).
- The handoff takes a stronger stance than this brief: it says the convergent-loss null is a finding, the modquad separation is proved/two-sided, and prior no-go statements often reflected confounded comparisons (`docs/HANDOFF.md` lines 174-181, 660-695). For Frontier, treat that stance as a prior-agent interpretation to test against new debug and extended evidence, not as an allocation authorization.
- Prior no-go statements about E97 or Emender should therefore be recorded as hypotheses or risk recommendations. They are useful warnings about kernels, ROCm risk, long-horizon recipes, and capability uncertainty, but not settled launch criteria without Frontier-specific evidence.

## Current hypotheses to test

1. `emender-mlp` and `gdn2-mlp` remain loss/perplexity-close under the Frontier debug recipe on commapile; any apparent short-run BPB ordering is unstable until replicated across seeds/checkpoints or visible outside the known noise band.
2. The `e97` versus `e97-linear` difference on length-extrapolated nonlinear state-tracking persists under the actual debug harness and is specifically tied to per-step nonlinearity-in-time, not capacity, optimizer, or kernel differences.
3. `e97-linear-MLP` should behave more like GDN2 on the nonlinear-in-time separator while retaining the MLP-controlled LM comparison; this arm is important if the implementation supports it because it isolates the `tanh` state map under the same channel mixer.
4. The ROCm E97 fused kernel can match CUDA/reference correctness within the stated bf16 tolerance and produce stable gradients/loss at T up to 2048 on a single MI250X GCD.
5. `gdn2-mlp` is the fallback systems arm if E97 ROCm parity or throughput fails, but that fallback should be justified by measured Frontier kernel evidence, not prior no-go language alone.
6. Plain-average DiLoCo with `outer_beta=0`, K=250, and the finalized inner optimizer remains in-basin on Frontier small-node debug runs; large-island drift may appear because the repo evidence stops at I <= 4/6.
7. AdamW plus warmup plus cosine is the safer long-horizon recipe for extended runs unless a Frontier debug run directly validates Schedule-Free behavior under the chosen merge path.
8. Any capacity differences that are invisible in short BPB runs may appear only in the >100B-token long tail: real-LM state-tracking/reasoning deltas, large-island DiLoCo drift, and emender-vs-gdn2 capability divergence should be tracked at fixed token intervals rather than decided from early loss.

## Debug-queue evidence required before expensive extended runs

Do not recommend extended-queue launch packages until the debug queue has produced:

- ROCm/HIP audit outcome for E97, e97-linear if used, and GDN2 kernels, including unsupported Triton features, bf16 assumptions, and fallback paths.
- Single-GCD correctness parity for the E97 fused split-edit kernel against CUDA/reference or an accepted CPU/PyTorch reference, including T = 128, 512, 1024, and 2048.
- Single-GCD and small-node throughput for `emender-mlp`, `e97-linear-MLP` if available, and `gdn2-mlp`, including tokens/s, memory, max batch, compile failures, NaNs, and exact git commit.
- Short commapile loss sanity for each arm with the intended AdamW/warmup/cosine recipe, enough to catch recipe collapse, NaNs, or tokenizer/data-path errors.
- A minimal length-extrapolation/capability sanity panel that includes `e97`, `e97-linear`, and `gdn2` or their MLP-wrapped equivalents where technically meaningful.
- Hierarchical DiLoCo smoke evidence on Frontier small nodes: merge correctness, consensus checkpoint restart, `outer_beta=0`, K=250, RCCL/aws-ofi-rccl configuration, and observed matched-token loss behavior.
- Allocation ledger entries for node count, walltime, job ids, actual node-hours, and projected extended-run cost.
- Confirmation that commapile is staged, decompressed or streamable as planned, integrity checked, and legally the data used by the releasable path.

## Claims requiring Frontier evidence before launch criteria

- "E97/Emender is the launch arm" requires Frontier ROCm kernel parity, throughput, loss sanity, and at least debug-scale capability evidence. The handoff supports testing it, not launching it blindly.
- "GDN2 is the launch arm" also requires evidence: GDN2 may be the more mature fallback, but prior no-go language for E97 is not enough unless E97 fails a cited debug gate.
- "e97-linear-MLP is sufficient" or "the nonlinearity is unnecessary" requires a direct e97-linear/e97 comparison under matched recipe and relevant length-extrapolated tasks.
- "Loss/perplexity decides the architecture" is not supported by the paper or handoff; both describe loss convergence/ties. Loss can be a sanity and regression metric, not the sole launch criterion, unless a run shows a large, replicated loss failure.
- "Capacity differences are absent" cannot be concluded from <=2B-token or debug runs. The handoff's central open question is whether capability divergence emerges around the long-token regime (`docs/HANDOFF.md` lines 698-706); >100B-token tracking is needed.
- "DiLoCo scales to 512 GCDs/islands" requires Frontier small-node and then larger-node evidence. Local I <= 4/6 results do not settle the large-island regime (`docs/HANDOFF.md` lines 431-437, 455-461).
- "64 nodes yields 150-180B tokens/day" is an external planning estimate until P1 single-GCD throughput and current OLCF policy are verified (`docs/HANDOFF.md` lines 505-524).
- "commapile upload path is ready" requires confirmed bucket/prefix and integrity hash; the handoff marks the S3 path as an operator directive needing confirmation (`docs/HANDOFF.md` lines 543-548, 760-764).

## Open questions

- What exact CLI/model name should represent `e97-linear-MLP` in the Frontier debug matrix, and does it share enough wrapper plumbing with `emender-mlp` to isolate state nonlinearity cleanly?
- Does the current ROCm stack support the required Triton features for `ndm/triton/e97_chunked.py` or the sequential fused E97 path without silent eager fallback?
- Which inner optimizer will the Frontier run actually use: Schedule-Free with the fixed merge semantics, or AdamW with warmup/cosine? The choice changes which merge risks must be tested.
- How should the debug capability panel be scaled so it catches nonlinearity-in-time regressions without overfitting to short synthetic runs?
- What threshold, if any, should convert debug evidence into a human-approved extended package? This brief recommends thresholds be based on parity/correctness, no NaNs, stable loss, and measured throughput/cost rather than a declared architecture winner.

