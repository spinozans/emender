# E97 private-analysis runtime and authority progress

Date: 2026-09-08

## Direction

This work implements the opt-in `e97-pi-agent-analysis-v1` branch without
changing legacy Pi-v2 serialization. It is a prerequisite to the matched
action-only/private-analysis representation comparison, not authorization to
start sustained GPU training.

## Runtime closure completed

- `ndm/e97_agent_protocol.py` implements canonical JSON-string `Analysis:`
  framing, strict parsing, bounded bytes, completion detection, and exact
  `reasoning_content` mapping.
- `ndm/e97_agent_server.py` binds analysis mode into serialization, generation
  completion, OpenAI responses, SSE, recurrent cache prefixes, and versioned
  external-service identity.
- Legacy external service identity remains
  `emender-e97-agent-service-attestation-v1` with no added fields.
- Analysis services use `emender-e97-agent-service-attestation-v2`, runtime
  identity v2, and `agent_protocol=e97-pi-agent-analysis-v1`.
- Analysis-mode external responses include a server-computed canonical assistant
  message SHA-256. The trusted controller rejects missing or changed assistant
  bytes before accepting the turn.
- `ndm/e97_acquisition_controller.py` preserves private analysis in messages,
  request identities, completion receipts, metadata transcript hashes, body
  bounds, tool turns, and finals. Legacy mode rejects the extra field.
- `ndm/e97_onpolicy_records.py` accepts exact v1 or v2 attestation schemas and
  validates the v2 assistant-message digest.
- `ndm/e97_moe_agent_server.py` and
  `scripts/serve_e97_moe_agent_openai.py` expose the same explicit completion
  mode. MoE workers consume the coordinator's completion decision.
- `configs/pi/e97-dense-agent-analysis.models.json` is a distinct Pi model
  identity with reasoning enabled; the existing legacy config is unchanged.

## CPU evidence

Targeted suite command:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest -q \
  tests/test_e97_open_swe_private_analysis_sft.py \
  tests/test_e97_agent_protocol.py \
  tests/test_e97_agent_server.py \
  tests/test_e97_onpolicy_records.py \
  tests/test_e97_acquisition_controller.py
```

Most recent combined protocol/server/controller/attestation/MoE/converter suite:
**92 passed**. These tests cover legacy isolation,
malformed and oversized analysis, delimiter collisions, visible-final isolation,
SSE mapping, cache replay identity, trusted-controller preservation, a synthetic
eight-action sequence, and post-service client mutation rejection.

Installed Pi system probe:

- script: `scripts/probe_e97_pi_analysis_roundtrip.py`;
- receipt:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi_private_analysis_roundtrip_probe_v1.json`;
- receipt SHA-256:
  `d8c56c26899819482d7cc664bc006e13548176dd0886ad2d09c9a151b8afb4e1`;
- source-binding attestation SHA-256:
  `87b880477c281c5fb4f36f5bc0572ac6498bbf1e07346a7aa3095db3634285fa`;
- probe source SHA-256:
  `dc0e8a3424991746b2da1b28ab373fe604d88447e6401627d5f8e3e9dfcdfe1e`;
- result: 8 streamed reasoning/action/read/observation turns plus one final;
- 9 HTTP requests; recurrent cache was one miss followed by eight hits;
- all reasoning hashes round-tripped exactly;
- this is CPU protocol evidence, not useful-model inference.

## Cap decision

The complete-trajectory cap audit is recorded in
`docs/validation/e97-open-swe-private-analysis-cap-audit.md`.

Artifact:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open_swe_reasoning_cap_trajectory_audit.json`

SHA-256:
`5b536a4ec4a2225ace2c75448acecc38417e5e90c45421c4fae8f577ae8c278e`

Source-binding attestation SHA-256:
`d394ff2f728ed1b258fade5d31f788eb45e226b2303f6190233b84e0cd7e167f`

Decision: provisional 2,048 `p50k_base` tokens and 65,536 UTF-8 bytes per
emitted analysis; exclude the complete trajectory on overflow. This excludes
60 of 14,314 action-normalizable trajectories and never truncates reasoning.

## Matched authority implementation

`scripts/build_e97_open_swe_private_analysis_sft.py` builds either:

- `--representation private-analysis`; or
- `--representation action-only`.

Both representations run through the exact same private-analysis admission gate,
including source filters, the 2,048-token/65,536-byte analysis bounds, and the
private 64K complete-unit segmentation check. This prevents an overflow from
leaving a trajectory in only one arm. The action-only branch removes only a
validated canonical `Analysis:` frame after admission. Each authority emits a
complete per-source trajectory receipt and snapshots the actual builder,
normalizer, tokenizer codec, protocol, and dataset source bytes.

A 100-source canary produced 25 included trajectories and exactly matched their
identities across the two arms. Private analysis had 338,517 assistant target
tokens; action-only had 195,312. Both preserved 1,321 target logical units. The
difference establishes why final experiments must match consumed assistant-target
tokens in addition to trajectory identities.

Two pre-authority attempts are retained but invalid: v1 was stopped after a
builder edit; v2 was stopped after finding that auxiliary/source snapshots in
the core `outputs` map violated the existing masked-SFT consumer contract. Both
are incomplete, non-authoritative staging directories and have not been deleted.

The immutable-source-snapshot private build completed successfully:

- manifest:
  `/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-private-analysis-64k-v3/manifest.json`;
- manifest SHA-256:
  `248a02e6d977b83474eba6e48589a691e4fc36115602c931cfac4960b73dc196`;
- builder source SHA-256:
  `0246c8aa123192ab6ae7de08cc3cdd202fcfd2fb7f9909266cd219673813093f`;
- 10,905 included complete trajectories;
- 13,112 records and 558,858 targeted logical units;
- 564,969,738 total tokens;
- 151,602,474 assistant targets, including 59,507,270 private-analysis tokens;
- 3,982 source-local `think` turns folded into the following executable analysis;
- core `outputs`: exactly tokens, mask, record index, and metadata;
- trajectory receipts and source snapshots: separately hashed manifest fields.

The strict converter excluded 3,342 trajectories containing empty or `C-c`
`execute_bash` turns that cannot be represented faithfully by the stateless typed
bash interface. It also excluded 70 trajectories after folding `think` reasoning
caused the emitted 2,048-token analysis cap to be exceeded. These are complete-
trajectory exclusions, not truncations.

The matched action-only build also completed successfully:

- manifest:
  `/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-matched-action-64k-v3/manifest.json`;
- manifest SHA-256:
  `f1226ebc6f74cbd528887ff742d28d4a65ea42207d02e676e9dc849cda961726`;
- identical builder source SHA-256:
  `0246c8aa123192ab6ae7de08cc3cdd202fcfd2fb7f9909266cd219673813093f`;
- 10,905 included trajectories and 558,858 target logical units;
- 12,215 records and 497,630,615 total tokens;
- 87,275,335 assistant target tokens;
- exclusion counts exactly match the private-analysis manifest.

Full cross-authority validation passed:

- receipt:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/open_swe_matched_representation_validation_v3.json`;
- receipt SHA-256:
  `08541a03dc041e91cc4317209537ee735928c2d5406a760533aaace4b802101d`;
- identical included trajectories: 10,905;
- included identity-set SHA-256:
  `b993eb9b4fc013074f9d0f6eeacf8763455ea73428abecbd58cce0769e6f2a81`;
- identical target logical units: 558,858;
- all output hashes, source snapshots, receipt statuses, record indexes, masks,
  protocol parses, analysis limits, token totals, and target totals passed.

Boundary-aware diagnostic packs completed and validated:

- private pack manifest SHA-256:
  `89a08ce5dbdf42189decb8921a1d373d465dbc998eca8d489478c8e1b43a672c`;
- private: 11,558 packs covering all 13,112 records with zero oversize exclusions;
- action-only pack manifest SHA-256:
  `bf7b3f3db5c7191be1c1a94aea39742f01b23972629543cd9891232c88d9e39c`;
- action-only: 10,325 packs covering all 12,215 records with zero oversize exclusions;
- context: 65,536 plus the prediction-aligned next token;
- sampler identity: `epoch-permutation`;
- reset, cross-document masking, padding invariance, protocol, and dataset tests:
  46 passed;
- combined pack-validation receipt:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/open_swe_matched_pack_validation_v3.json`;
- receipt SHA-256:
  `9b3745f844896792dac2ff79a2db58d52ce9fc61d48616c017e883c67e7bc4ee`;
- packs remain explicitly non-trainable candidate evidence; no GPUs were used.

Next:

1. design a deterministic token-matching schedule that retains the same unique
   trajectories while accounting for the approximately 1.737x target-token difference;
2. freeze the development comparison recipe, holdouts, LR, and stop rules before
   any numerical GPU canary.

## Remaining blockers

- Full matched authorities and boundary-aware packs validate; the token-matched comparison recipe remains unfrozen.
- Analysis evaluation identities must declare at least 4,096 completion tokens;
  legacy 512-token service/controller limits do not cover the 2,048-token
  analysis source envelope.
- Fused-CUDA token-delta versus replay numerical qualification is still open.
- Source mixture, source caps, target-token downsampling, LR schedule, milestones,
  regression floors, and stop rules are not frozen.
- No sustained GPU training is authorized.
- The checked first-party registry remains candidate-only and unrelated to this
  public-source representation experiment.
- `tests/test_serve_e97_agent_openai_identity.py` currently passes 9/10. The
  remaining test fails closed with `loaded replay controller code does not match
  verified archived source`, because the controller was intentionally modified
  after the old source archive was sealed. This is expected identity evidence,
  not a private-analysis behavior failure. Do not weaken the test; a new reviewed
  source archive is required before formal first-party admission.
