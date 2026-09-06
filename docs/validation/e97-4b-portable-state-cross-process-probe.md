# E97 4B portable-state cross-process probe

## Verdict

A compact recurrent state from the 4B E97 model was saved in one Python
process, the process exited, and a second Python process restored the artifact
and reproduced the same 32 greedy token IDs with **zero transcript-token
replay**. This qualifies the initial same-host, same-checkpoint, fused-CUDA
save/restore primitive. It does not qualify model answer quality, cross-revision
state compatibility, cross-machine numerical parity, or the planned 100-file
demonstration.

## Identity

- Model checkpoint SHA-256:
  `7d0e17afd6dde6b4846c2143580fc87d53f1e8a645bfb87acf1f3d2428e7cf37`
- Weight representation: saved Schedule-Free `x`
- Tokenizer: `p50k_base`
- System-prompt SHA-256:
  `8e53d2bc8f7603278e6fe18763b372769df848583c4d049f340c49bbc5054d26`
- Runtime: tokenwise fused CUDA E97
- Artifact schema: `emender-e97-portable-state-v1`

## Artifact

- Path:
  `/mnt/nvme2n1/erikg/e97_state_demo/terminal-x-cross-process-probe-v1/session.e97state`
- SHA-256:
  `88cc3bb933a836e382c3f528a8dc84f2dfbdf50adff901bb1a28cb64978535f8`
- Artifact bytes: 18,206,108
- Recurrent tensors and boundary logits: 17,795,282 bytes
- Mode: unencrypted diagnostic
- Permissions: `0600`
- Transcript tokens included: no
- Source prefix token count: 51
- Token-lineage SHA-256:
  `0a81177f0fd260df7a37208aed013369e8205be197326415557e5d76aa0b2e58`

The artifact is about 17.36 MiB on disk and has size independent of transcript
length because it contains recurrent tensors, boundary logits, bounded
metadata, and integrity framing rather than the transcript.

## Process evidence

Creator receipt:

`/mnt/nvme2n1/erikg/e97_state_demo/terminal-x-cross-process-probe-v1/create.json`

- model load: 30.658 seconds
- 51-token initial ingestion: 2.060 seconds
- 32-token generation: 0.854 seconds
- replayed/initially ingested prefix tokens: 51
- creator process exited with code zero

Restore receipt:

`/mnt/nvme2n1/erikg/e97_state_demo/terminal-x-cross-process-probe-v1/restore.json`

- model load: 30.526 seconds
- artifact verification and device restore: 0.149 seconds
- transcript tokens replayed: **0**
- 32-token generation: 1.497 seconds
- exact greedy token parity: **passed**
- restore process exited with code zero

The restored continuation matched all 32 expected token IDs and decoded bytes.
The generated content itself was an irrelevant fabricated `read` action, which
is consistent with the separately documented behavioral collapse of this
checkpoint. This probe validates state mechanics only.

## Fork and branch-isolation evidence

A third process restored the parent artifact with zero replay, deep-cloned it
twice, and applied distinct token deltas:

- branch A: 20 delta tokens; terminal token count 87;
- branch B: 18 delta tokens; terminal token count 85;
- parent tensors and logits remained bit-exact after both branches: passed;
- parent reference continuation after both branches: exact greedy parity passed;
- branch recurrent tensors and token lineages differed: passed;
- branch A artifact SHA-256:
  `0d6d1463d8a5c98e7cd8364d99b6baf34eecb6682a11fd8c4c0c39e93886bd74`;
- branch B artifact SHA-256:
  `c26ed28db61a24ad0a91b36cbf87388c172f7686f2fbcc1d9f652a691cafc045`.

Both collapsed-model branches generated the same first 16 tokens despite their
different hidden states. Therefore this passes storage isolation and unchanged
parent semantics, but does not demonstrate useful behaviorally distinct
answers. That remains a model-quality gate.

Receipt:

`/mnt/nvme2n1/erikg/e97_state_demo/terminal-x-cross-process-probe-v1/fork.json`

## Security and compatibility properties

The implementation:

- binds restore to exact model checkpoint SHA-256, tokenizer identity,
  system-prompt SHA-256, and runtime schema;
- verifies an internal SHA-256 integrity trailer before deserialization;
- loads only a tensor/primitive `torch.load(..., weights_only=True)` payload;
- supports optional AES-256-GCM encryption with a caller-provided 32-byte key;
- writes atomically, fsyncs the artifact and parent directory, and applies mode
  `0600`;
- rejects compact state in the full-history prefix-reuse path, requiring
  explicit token-delta continuation rather than silently replaying or assuming
  transcript compatibility;
- supports deep tensor cloning for isolated branches.

Recurrent state is sensitive session data. Unencrypted artifacts must not be
published or moved to an untrusted location. States from different checkpoint
revisions are incompatible unless a future qualification proves otherwise.

## Automated validation

The portable-state, server, protocol, and Pi integration selection passed:

```text
34 passed
```

Covered cases include compact round trip, tensor equality, artifact corruption,
model and prompt identity mismatch, optional encryption and wrong-key failure,
clone storage isolation, server suffix caching, transactional commit/rollback,
and protocol validation.

## Remaining gates

1. Add an explicit token-delta HTTP/session API; compact restored states must not
   depend on a full transcript request.
2. Demonstrate behaviorally distinct and useful branch answers; state-level
   suffix divergence and unchanged-parent semantics have passed.
3. Run create/restore using the behavior-selected parent as well as the
   terminal diagnostic checkpoint.
4. Transfer an encrypted artifact between machines with the same immutable
   checkpoint and compare declared numerical behavior.
5. Ingest roughly 100 complete files, terminate, restore, and fork multiple
   analyses while reporting model load, state restore, replay count, HBM, state
   size, and branch latency.
6. Compare measured state and serving costs with a matched Transformer KV-cache
   baseline without claiming equivalent model quality.
