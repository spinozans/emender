# Analysis runtime candidate qualification

## Scope

Evaluation-only candidate derived from training source archive
`57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4`.
This does not update the canonical first-party registry, authorize collection,
or promote either trained checkpoint. Existing immutable training sources are
unchanged. R16 evidence discipline applies by analogy; no elastic/native/async
DiLoCo conformance is claimed.

Candidate root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/analysis-runtime-candidate-v1`

## Source identity repair

The prior serving launch failed before loading a model because its loaded
controller differed from the retained controller archive. The repair rebuilds
the closed, allowlisted source archive from actual source bytes **inside the
isolated candidate only**, then executes the unchanged loaded-code verification.
No skip flag, exception suppression, or legacy runtime substitution is used.

Reproduction:

```bash
OUTPUT_ROOT=/new/absolute/candidate/path \
  bash scripts/prepare_e97_analysis_runtime_candidate.sh
```

The recipe is local to the pinned source archive path named in that script.

Evidence:

- `source-binding.json`: SHA-256
  `2aad3e067b2ee7ef7f8becbd22d4666427eaaf774de76b78b1d1949151a98807`
- CPU tests: **115 passed**, including serving identity **10/10**, source
  snapshots, protocol, dense/MoE service wiring, controller, on-policy records,
  and matched converter tests.
- `runtime-tests.xml`: SHA-256
  `cba16fedc443b796f34d4d4d6c5dcaf97a93e251a2ac745c1d0812f0b0c7bac5`
- Candidate file hash inventory including the new CUDA probe:
  `evaluation-source-files.sha256`, SHA-256
  `e4d2a266e19f801d636d69a0b9c5ed7daaab79f3f8d57dfe3ef06114a3cb215e`

## CUDA gate (passed for the bounded dense tokenwise fixture)

The v1 GPU probe stopped at the first tool boundary on an incorrect probe
assertion: v1 token lineage hashes include the delta length and previous lineage,
so equal token sequences ingested with different call boundaries do not have
equal lineage hashes. State/logit comparisons at that boundary had passed.
No runtime source or numerical tolerance was changed. The corrected probe
compares complete token histories/counts/checkpoint identity and independently
verifies each delta lineage against its own call history. Two regression tests
pass, including rejection of tensor, token, counter, checkpoint and lineage
changes. The earlier launcher error and probe failure logs remain in v1.

The corrected candidate is `analysis-runtime-candidate-v2` under the same parent
root. Its file-inventory SHA-256 is
`07e276b2a074d0661d1c58e6aa034b84271b7279d3b9261a893887fe29eefc25`.
Only the probe and its new tests differ from v1 (excluding cache files).

`run-cuda-probe.sh` runs `scripts/probe_e97_analysis_cuda_replay.py` against
private-trained checkpoint
`6881acf1d79f60ea910752277ac4809bdd39d6d7598d47489c7e9bc3a1df6fcd`,
sequentially for explicit `saved` and `train` weight modes. GPU 0 performs the
probe; GPUs 0–7 are exclusively reserved to prevent competing jobs.

Predeclared gate: all nine prefix boundaries must have finite, bit-identical
recurrent tensors and next logits between cached tokenwise ingestion and
full-prefix tokenwise replay; eight greedy continuation tokens must match.
Input state must remain unchanged after generation. Fixtures contain eight
scripted tool exchanges and private analysis with Unicode, newlines and embedded
protocol markers. They are numerical fixtures, not task-success evidence.

Both weight modes passed all nine boundaries (236–804 prefix tokens), including
bit-identical state/logits, equal eight-token greedy continuations and independent
verification of the boundary-sensitive delta lineages:

- `analysis-runtime-candidate-v2/cuda-replay-saved.json`: SHA-256
  `e46a872f23b1c7e135677ca0281d0cfb715374f559cbfd633c9c2af6f49888d4`
- `analysis-runtime-candidate-v2/cuda-replay-train.json`: SHA-256
  `ca513079ed9584d18fa60122b56a2af8b24aa93eaeaf132d227eda1a28f1bb57`

This is a short-prefix numerical check, **not** a 64K numerical qualification.
It does **not** qualify segment ingestion, MoE CUDA behavior, real HTTP/Pi
round-trip on the 4B model, cross-machine state transport, or capability gains.
Those remain separate gates. Runtime candidate is not yet release-qualified.

## HTTP plumbing gate (blocked on generated protocol)

A local, one-hour-bounded analysis-enabled service on port 24980 uses explicit
train/y weights, 4,096 output tokens, tokenwise ingestion, external-controller
mode, and the byte-verified sandbox image. A read-only two-file fixture client
checks v2 attestation, server-bound assistant digests, exact reasoning round-trip,
cache miss/hit progression and completion-token bounds. Unknown calls/paths
are rejected without dispatch. This diagnostic client is not the formal
collection controller and its results are not training authorities or holdouts.

The first real request with train/y weights was rejected with HTTP 422:
`analysis turn requires a canonical Analysis JSON string`. No assistant turn
was admitted, no tool was dispatched, and no HTTP round-trip pass exists.
Artifact: `analysis-runtime-candidate-v2/http-roundtrip-train-v1.json`.
The identical request was replayed with bounded generated-error tracing only.
It again failed before dispatch: output began with legacy `Action: read`, named
an unrelated `read_and_write.py` path, fabricated observation text and repeated
additional actions. No `Analysis:` frame preceded it. The trace receipt
`http-roundtrip-train-traced-v1.json` has SHA-256
`4d50ce513745fe3f81be1fa06df366e35366151cb97739a423c71806fe3d17a9`.
This is generated-protocol/grounding failure on a diagnostic fixture, not a
reasoning transport pass or a representative capability score. The parser was
not relaxed. Saved/x also failed on the same request: the same unrelated
legacy read call was followed by repetitive prose, again without an analysis
frame. Its receipt `http-roundtrip-saved-traced-v1.json` has SHA-256
`e4d979d4679c58ec3839dede6b23d8ecbbd7623912ceb17263440f9f004c0648`.
Both services have been stopped. Neither weight mode passed the live HTTP gate;
neither checkpoint is promoted. CUDA state parity does not establish that the
model emits the protocol correctly.

### Read-only serialization audit

A synthetic Unicode/embedded-marker fixture passed exact training-versus-serving
byte and token equality and assistant-target-start checks for a tool and final
turn, in both the working checkout and isolated candidate. This is one fixture,
not an audit of every training trajectory. The test artifact is
`test_e97_analysis_serialization_parity.py`, SHA-256
`496e1edeeb6a0b9ac96bac4d74da826a26d0d8c0ae7d3f0f66051b457c610cfb`.
The aggregate scan of the sealed private authority passed after verifying all
four payload hashes. All 1,307 agent records have the exact analysis system
prompt; all 55,396 analysis target-run starts follow the assistant header.
First-target prefix tokenization matches exactly in all 1,307 records. The
remaining 221 target-run starts are standalone record separators. Shared-source
targets remain unchanged; agent targets are 15,002,508 of 50,008,637 authority
targets. This rules out the specific missing-frame/masked-start hypotheses at
these checked boundaries, not optimizer efficacy or full model correctness.

Receipt `training-analysis-boundaries-v1.json`: SHA-256
`6e8c0ba25e29c7d46b137c53defb4bdfb1b20cb21bc4817050dafe6232fb6601`.
The first invocation finished computation but could not publish because the
output used `../`, correctly rejected by atomic publication. The successful
rerun used an absolute output path; no check was bypassed.

**Current verdict:** numerical/cache and bounded serialization checks pass;
free-generation analysis protocol on the live diagnostic fails in both x/y.
Do not conflate that rejection with an HTTP transport bug, or infer a broad
capability score from this one fixture. No further training, promotion, or
prompt-adaptive rescue was performed. The broader fresh capability panel and
independent final holdout remain unfinished.

The original client bytes were retained as `probe_e97_analysis_http_v1.py`,
matching the receipt's source SHA-256
`fc0322c7903529212fae5a680ba98a6aee7d98ddc470229d9a4731a4cc4e9c76`.
The explicit-weight-mode client is `probe_e97_analysis_http_v2.py`, SHA-256
`1857efda4fa8f745b17bc892213e3bca545814fffcac22a3e661dfacf5d51073`.

The initial server teardown required explicitly terminating the verified
`timeout` child process group after the process manager's stop timed out.
All three launcher/timeout/server PIDs were confirmed gone and GPU allocations
released. The trace launcher uses `timeout --foreground` so its children remain
in the managed group. Deliberate teardown is not a model/runtime crash.
