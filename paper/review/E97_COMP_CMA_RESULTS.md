# E97 comp-CMA results - recovered artifact

**Task:** `e97-comp-cma` (agent-1207)
**Date of original run:** 2026-06-07
**Date recovered:** 2026-06-08
**Status:** recovered from surviving WG task logs and downstream committed synthesis; raw CMA
JSON / the original markdown deliverable did not survive the cleaned worktree.

## Source status

This file is the missing committed artifact for the `e97-comp-cma` run. It does not
pretend that the original markdown or raw candidate JSON survived. I searched the
current tree, WG output, surviving `/tmp/claude-1001/...agent-1207...` task outputs,
and sibling WG worktrees for `E97_COMP_CMA_RESULTS.md`, `g2_i2`, `g2i2`, `g2_i6`,
`3.203`, `3.377`, and comp-CMA-specific outputs. The surviving primary record is
the `wg show e97-comp-cma` log, mirrored in `.wg/output/e97-comp-cma/log.json`.
The downstream committed docs `E97_WITHIN_LAYER_SYNTHESIS.md` and
`E97_SCALE_PILOT_RESULTS.md` corroborate the same interpretation.

Therefore the numeric values below are real recovered run outputs, not mocks. Where
the WG log retained only a qualitative sign or conclusion, the table says "not
preserved" instead of inventing a value.

Primary recovered log line, 2026-06-07T22:31:35Z:

> CMA search COMPLETE: 26 evals (23 valid, 3 diverged=i0/gen NaN w/ extreme knob-LR). Fraction-ablation: ONLY e97_raw (LM backbone, -0.34 bpb corr) + gdn2_recall (recall/track/nonlin) earn place; latch(+0.87 bpb!), nonlin(+0.60), count, e97_delta are liabilities backbone already covers. Combined winner g2_i2: 38gdn(plain)+9raw, knob23, bpb 3.203 (beats all prior cells) recall0.94 no-track. Cap-complete g2_i6: 24gdn-neg+19raw, bpb3.377 cap0.919.

## Search protocol

| field | recovered value |
|---|---|
| worktree / code guard | `HEAD=cd10853`; `e97_raw` and `e97_delta` fused head types present in `typed_head_mixture.py` |
| kernel guard | `verify_e97_within_layer_heads.py` passed: no eager fallback; fused path active |
| smoke check | 1.5 min LM screen printed `FINAL_HELDOUT_BPB=3.6194`, 148,388,160 params, fused `typed-gdn2-lm` |
| optimizer | `cma 4.4.2`, popsize 8, 3 generations |
| search space | 9 dimensions: 6 head-type fraction logits plus `gdn_allow_neg`, `knob_lr_mult`, `mlp_ratio` |
| head types | `gdn2_recall`, `count`, `latch`, `nonlin`, `e97_raw`, `e97_delta` |
| fitness | held-out BPB minus `0.4 * norm_capability` over 5 probes |
| LM screen budget | 12 minute screens during CMA |
| probe budget | 5 probes at 3500 steps during CMA |
| controls | `dense_gdn2_mlp` and `pure_e97_delta` were included at the same screen budget |
| completed evaluations | 26 total, 23 valid, 3 diverged |
| divergence cause | initial/gen NaNs under extreme knob learning-rate settings |
| confirmation caveat | a later full-budget confirmation launch had CPU contention; the original CMA driver ran LM phase then probes sequentially, so the search results were logged as valid |

The post-CMA confirmation attempt was explicitly marked non-apples-to-apples because
concurrent probe batteries starved the E97 Triton LM screens of CPU (`13%` GPU util
versus `98%` for pure GDN). That affects the abandoned confirmation phase, not the
search table below.

## Fraction-ablation / per-head value

Lower BPB is better. Negative BPB correlation means a head type helped the LM screen;
positive BPB correlation means adding that head type correlated with worse LM BPB.

| head type | recovered BPB association | earns place? | reading |
|---|---:|---|---|
| `e97_raw` | `-0.34` BPB correlation | yes | The LM backbone head. It is the only E97 split-edit head with a recovered negative BPB association. |
| `gdn2_recall` | exact BPB coefficient not preserved | yes | Earns place through recall / track / nonlin capability and appears in both recovered operating points. Plain GDN is LM-best; negative-eigenvalue GDN is capability-complete. |
| `latch` | `+0.87` BPB correlation | no | Strong LM liability. The backbone already covers latch-like probe behavior, so standing latch heads are redundant for LM. |
| `nonlin` | `+0.60` BPB correlation | no | LM liability. The CMA did not reward a dedicated nonlinear head as a standing LM component. |
| `count` | liability; exact coefficient not preserved | no | Redundant for the LM composition because the backbone already covers count. |
| `e97_delta` | liability; exact coefficient not preserved | no | Did not earn a place despite being a fused head type and despite the pure-control screen. |

Result: the per-head-value claim is not "all six specialists are useful." It is
much narrower: only `e97_raw` plus `gdn2_recall` survived composition pressure.
`latch`, `nonlin`, `count`, and `e97_delta` should not be cited as useful standing
LM heads from this CMA.

## Operating points

| id | role | recovered allocation | GDN sign | knob LR | held-out BPB | capability / probe status | interpretation |
|---|---|---|---|---:|---:|---|---|
| `g2_i2` | LM-best combined winner | `38 gdn + 9 raw` as logged | plain GDN | `~23` (`23.4` in scale-pilot carry-forward) | `3.203` | recall `0.94`; no track | Best LM screen found by the comp-CMA. It beat all prior small cells on held-out BPB, but it is track-blind because it uses plain GDN rather than negative-eigenvalue GDN. |
| `g2_i6` | capability-complete operating point | `24 gdn-neg + 19 raw` as logged | negative eigenvalue allowed | not preserved | `3.377` | aggregate capability `0.919`; capability-complete operating point | Trades LM BPB for the negative-eigenvalue GDN mechanism needed for track. |

The logged head-count summaries do not preserve the full largest-remainder assignment
for every one of the six head types; they preserve the dominant nonzero heads. The
scale-pilot document later carried `g2i2_cmawin` as the exact comp-CMA LM-best
6-head-type mixture with `knob_lr 23.4`, which corroborates that the underlying
logits existed during downstream work even though the raw file is absent here.

The direct LM cost of choosing the capability-complete recovered point over the
LM-best recovered point is:

| comparison | delta BPB |
|---|---:|
| `g2_i6` capability-complete minus `g2_i2` LM-best | `+0.174` BPB |

That is the central operating-point split: `g2_i2` is LM-best and recall-strong but
track-blind; `g2_i6` is capability-complete but worse on LM BPB.

## Controls and corroboration

The surviving log says the CMA protocol included `dense_gdn2_mlp` and
`pure_e97_delta` controls at the same screen budget, but it does not preserve their
exact BPB rows. The conclusion is still constrained by later committed documents:

| committed artifact | corroborating point |
|---|---|
| `E97_WITHIN_LAYER_SYNTHESIS.md` | Reconstructs the same comp-CMA per-head finding: only `e97_raw` and `gdn2_recall` earn place; `latch`, `nonlin`, `count`, and `e97_delta` are LM liabilities. |
| `E97_SCALE_PILOT_RESULTS.md` | Carries `g2i2_cmawin` as the exact comp-CMA LM-best 6-head-type mixture with `knob_lr 23.4`; reports it as a scale fidelity check, not the practical recommended cell. |
| `E97_WITHIN_LAYER_STUDY_RESULTS.md` and audit docs | Independently show the capability split: `e97_raw` covers count/latch but is recall/track blind; adding GDN recall heads recovers recall, and `gdn-neg` is needed for track. |

## Citation-ready conclusion

Do cite:

1. The comp-CMA's per-head value result is sparse: `e97_raw` is the LM backbone
   (`-0.34` BPB correlation recovered), and `gdn2_recall` is the only other head
   family that earns composition weight.
2. Dedicated `latch` (`+0.87` BPB), `nonlin` (`+0.60` BPB), `count`, and
   `e97_delta` heads are LM liabilities in this CMA; they are redundant with
   capabilities the backbone already covers.
3. The LM-best operating point was `g2_i2`, logged as `38gdn(plain)+9raw`,
   `knob~23`, held-out BPB `3.203`, recall `0.94`, and no track.
4. The capability-complete operating point was `g2_i6`, logged as
   `24gdn-neg+19raw`, held-out BPB `3.377`, aggregate capability `0.919`.
5. The tradeoff is structural in the recovered results: capability-completeness
   via `gdn-neg` costs about `+0.174` BPB versus the LM-best plain-GDN point.

Do not cite:

1. A full numeric per-head coefficient vector. Only three coefficients/signs were
   preserved exactly in the logs: `e97_raw=-0.34`, `latch=+0.87`, `nonlin=+0.60`.
   `count`, `e97_delta`, and the exact `gdn2_recall` BPB coefficient were not
   preserved.
2. The abandoned full-budget confirmation BPB as apples-to-apples evidence. It
   was explicitly invalidated by CPU contention in the log.
3. The six-head-type `g2_i2` mixture as the practical scale recommendation. The
   scale pilot found the exact 6-head-type carry-forward too kernel-fragmented to
   reach a usable step count at scale. The practical capability cell remains the
   simpler `e97_raw + gdn-neg + MLP` mechanism result, while `gdn2-mlp` remains the
   better scale LM cell.
