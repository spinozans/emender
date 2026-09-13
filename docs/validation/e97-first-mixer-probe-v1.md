# E97 first-mixer component probe v1

The [complete matrix reproduction](e97-activation-alignment-v1.md) now binds
actor and trainer endpoints exactly and repeats exactly. Its first measured
sequence-layout divergence is layer0 mixer output. That boundary includes input
normalization, projections, nonlinear transforms, recurrence, gating and output
projection; identifying it is not a numerical fix.

## Frozen diagnostic

Same four previously inspected turns and correction-y checkpoint. Four existing
profiles (`actor-cache`, `full-eval`, `masked-padded-eval`, `training-default`) x
four turns x two immediate repeats = **32 evaluations**, one fresh GPU worker.
Each profile's probabilities must bind within .0001 to the corresponding profile
in the successful matrix reproduction. Require identical full component-bank
hashes/dtypes and logp difference <=.0001 on repeats. Interpret components only if
both reference binding and repeatability pass. No threshold relaxation.

Reference measurements SHA:
`8336d6d5c21b4b4746a44d9e15d7b8978f5a910b3a25ea511257a21c862d02fa`.
Reference summary SHA:
`04f42be946e9226152e18d18bde252a1a451062745a728e91da642ec49866594`.
Source matrix recipe SHA:
`1aebe6302d42032ca5f9a0af3d63cc5f36d7f401b509801d1a483952cb23d9d5`.

Read-only module hooks capture thirteen first-layer boundaries:

- wrapper input and **actual normalized mixer input**;
- QKV projection input and separate raw Q/K/V outputs;
- raw alpha, output gate, erase gate and value-write gate projections;
- output-projection input/output and final mixer output.

Capture **every real causal input row**, including the prefix, not just prediction
rows. Track each wrapper call and each projection's chunk offsets separately;
exclude padding and the unused final consumed-token prediction. Record projection
chunk lengths. Current projection chunks are512 tokens. Real prefix lengths are
2,329/2,519/2,504/2,939, so final projection batches differ between segmented and
full-sequence layouts. Whether that changes projection values must be measured.

The raw projections precede their SiLU/sigmoid/decay transformations. The
output-projection input includes recurrence, nonlinear transforms and gating;
it is **not a bare recurrence output**. No kernel monkeypatch or output/activation
replacement is used; all hooks return None. The original prediction-row/head
observer remains active for reference binding.

Compare every component to actor-cache, retaining per-position absolute/relative
L2 differences, dtypes and hashes. 'First differing site' follows the declared
component order; parallel projection branches do not establish a strict causal
execution order. The primary actor/full-eval comparison is the existing segmented
versus full-sequence switch. Other selected profiles combine previously measured
settings and are not a new orthogonal factorial design.

## Bounds and validation

- `scripts/e97_first_mixer_probe.py`
- `scripts/diagnose_e97_first_mixer.py`
- `scripts/run_e97_first_mixer.sh`
- `tests/test_e97_first_mixer_probe.py`

Prefix<=4,096, generated<=512; at most4,607 observed real input rows, features
<=3,840 per component, one bank<=512MiB and six-bank accounting<=3GiB. Freeze
checks storage for the expected first-layer dimensions before GPU work; runtime
checks actual module types, dimensions, coverage, finiteness, unchanged parameters/buffers
and absent gradients. Persistent weights stay BF16; no FP32 master/optimizer.

Checked GPU lease, explicit local device/NUMA, isolated Triton cache, both
expandable allocator flags, hash seed0/Torch seed974223, highest precision/TF32
disabled and BF16 reduced-precision reduction disabled. Worker3,600s,
outer3,900s, kill grace30s. No retries or communicator reuse. Immutable exported
source checked before/after, including failure. Save each completed profile's
private receipt before continuing; never publish private generations.

CPU suite: **115 passed**. New tests cover complete-prefix/chunk/padding alignment,
unused final-step exclusion, actual mixer versus wrapper input distinction,
output/parameter preservation, missing modules, incomplete coverage and storage
failure cleanup. CPU tests do not qualify GPU numerics.

Real-input CPU freeze passed in `native-first-mixer-cpu-preflight-v2`; recipe SHA
`84a3195ad3277650a12398ae3e023b2bacebd6995ba1d5a53a4229fd13922e88`
is required before GPU acquisition. An earlier manual CLI invocation omitted
`PYTHONPATH` and failed at import, before freezing or GPU work; its failure is
retained in `native-first-mixer-cpu-preflight-v1/failure.json`. The launcher
explicitly exports the repository root as `PYTHONPATH`; no numerical code fix
was involved.

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-first-mixer-probe-v1`.
## Completed and independently audited

`proc_5040` completed in359s from `09a58f82`. All16 paired receipts (32
evaluations) matched their recorded hashes and aggregate measurements. Reference
probability differences and repeat probability differences were exactly zero;
all thirteen component banks repeated with identical hashes/dtypes. Parameter
and buffer fingerprints were unchanged, with no gradients or optimizer updates.
Peak allocated HBM was8,553,481,728 bytes. The frozen source inventory was also
independently reverified during the receipt audit.

Wrapper inputs, normalized mixer inputs and QKV inputs matched throughout every
real prefix. Nonetheless, BF16 projection outputs differed across layouts.
For task000/full-eval, Q/K/V first differed at position2,048; Q's maximum absolute
difference was.0625. Output-projection input first differed at2,048, but its
output already differed at position0 despite equal measured input there.

For task000/training-default, Q/K/V still first differed at2,048, while
output-projection input first differed at512. This is consistent with the
training/inference state-handoff precision difference, **not proof of its causal
sufficiency**. Output-projection input includes transformations and gating, not
bare recurrent state. The separately implemented
[FP32 state candidate](e97-fp32-recurrent-state-v1.md) is a controlled intervention;
this probe itself changed no numerics.

Summary SHA: `30e48d86d86124c883c0b177fd45dca25b6c1ff895e8ebb3f0dd6260fc4573c4`.
Measurements SHA: `3f85d83b3bb2a77693b9697ca03b5fc8e9d3122f000b0241b0fa6c234467a8a1`.
Independent `receipt-audit.json` SHA:
`3d5c83f26eac16e90fea061c370a28f5b31109f7071086b259e2fa0f1bfa37e6`.

No data admission, new rollouts or promotion. The original probability gate
remains failed and RL remains unqualified.
