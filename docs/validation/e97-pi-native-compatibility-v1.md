# E97 Pi-facing native compatibility candidate v1

## Authorization and scope

The operator approved the unchanged-weight Pi-facing adapter after the audited
representation-bridge experiment. This work authorizes no training, checkpoint
promotion, production installation, new source admission, or external service.
The bridge checkpoint remains SHA
`9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa`;
no checkpoint has yet been loaded for this adapter.

This is **Pi-fronted OpenHands**, not migration to Pi's `read/bash/edit/write`.
The [native protocol](../E97_OPEN_SWE_SOURCE_NATIVE_PROTOCOL.md) and the exercised
native executor remain authoritative. The older
[Pi runtime](../operations/e97-4b-pi-runtime.md) is a different model grammar and
has not been overwritten. See [learning status](../E97_AGENT_LEARNING_STATUS.md)
for the retained behavioral limitations and closed learning budgets.

## Implementation

- `scripts/e97_pi_native_bridge.py`: one authoritative `NativeEpisode`, exact
  full-history native prompt replay, public projection, checked call/result
  pairing, bounded generation, original executor routing, terminating finish.
- `configs/pi/e97-openhands-compat.ts`: explicit opt-in custom Pi provider and
  four native-name transport tools; no built-in host tools.
- `scripts/e97_pi_native_transport.py`: one offline isolated Pi CLI, private Unix
  socket, exact owned PID/UID peer check, bounded frames/deadline/request count,
  private receipts and owned child cleanup. No TCP listener, credentials,
  network model provider, fallback model, or retry.
- `scripts/qualify_e97_pi_native_transport.py`: authored controls through the
  real Pi CLI and qualified native sandbox, externally read filesystem oracle,
  independently reconstructed prompt bytes, error/privacy/cleanup checks.

The native model still sees the **original tool declarations and transcript**.
Pi sees transport envelopes, not a renamed native training trace. Executable
arguments cross JavaScript as a canonical JSON **string**, preserving large
integers, integral floats, invalid argument types and original executor error
semantics. Pi validates only that envelope. A reference binds each request to
its one pending native call. `think` exposes only that opaque reference; neither
its thought nor private analysis enters Pi's ordinary messages or tool UI.
Public commentary and final messages remain public. Native observations are
appended without rewriting, truncation or error synthesis.

Pi's finish acknowledgement terminates the tool loop (`terminate: true`); it is
not appended as an invented native observation and does not solicit an extra
model answer. The final persisted Pi history must match the entire expected
public projection before the owner accepts closure.

Only the isolated single-user/single-episode profile is supported. Loaded
project context, appended prompts, skills, images, arbitrary history edits,
extra users, compaction, branch/fork/resume, host `!` execution, and continued
conversation after native finish are not qualified. Unsupported paths fail
closed; this is not a general interactive deployment or persistent-session
qualification. No generic Pi system prompt is silently substituted for the
trained native contract.

## Evidence to date

Initial CPU attempts are retained:

- `proc_bbfe`: 16 passed, four failed. Unit tests caught an in-process returned
  call alias (now deep-copied); live Pi exposed an additional `label` field in
  its tool metadata, correctly rejected by the initial strict contract.
- `proc_8846`: 19 passed, one failed; the remaining failure was that metadata
  mismatch. The explicit expected transport declarations now include the label.
- `proc_e198`: **32 passed**, including a real Pi 0.85.1 CLI loop with authored
  replies, private-field exclusion and termination without another generation.
- `proc_7b52`: **70 passed** in the expanded regression run, including original native runtime
  and executor tests. Its live Pi control also checks lossless invalid native
  arguments, `2.0`, a >2^53 integer, and an error tool result across JavaScript.

`proc_199a` then passed the immutable `e1d15dd3` export and a copied Pi 0.85.1
runtime: another70 CPU tests, both real OpenHands controls, **14 exact full native
prompt checks,11 native calls,three real errors**, persistent shell state, both
external filesystem oracles, and two agent plus two reader cleanups. Source,
Pi runtime and interpreter before/after audits passed. A separate post-run audit
reconstructed all14 native turns/prompts/token sequences, checked actual Pi
message/dispatch counts and owned peer identities, rechecked both reader outputs,
private-field exclusion and allfour distinct cleanups.

Artifacts: `R/pi-native-compatibility-v1-control` and
`R/pi-native-compatibility-v1-scripted`, where
`R=/mnt/nvme2n1/erikg/e97_systematic_posttraining`. Summary SHA:
`9cdc588caaf749cc3ea9ee2d4b35188f43035d8d60d99f257607d7b5b3efa610`.

These are CPU/unit/scripted transport results, **not E97 capability evidence**.
Actual model evaluations and training updates remain zero at this point.

## Frozen next checks

1. Two authored seven-turn controls: persistent shell plus create/edit/readback;
   genuine missing-file, failed-shell and no-match errors followed by repair.
   Real Pi, original native sandbox, paused external filesystem reader. Every
   native prompt independently reconstructed byte-for-byte. No model sampling.
2. Next matched diagnostic: exactly eight reused fresh cases—the `0000` paired
   worlds in each of lookup/sum/edit/recovery—from the audited96-case panel.
   Sixteen episodes total, one GPU, unchanged bridge checkpoint, explicit
   live-y (`train`) mode, original greedy generator and budgets. Alternate route
   order by case. Require8/8 direct and8/8 Pi completions, per-case success parity,
   exact initial prompt identity, exact independent replay of every Pi native
   prompt, and verified Pi closure. Cross-run full transcript/token equality is
   reported descriptively, not made a blanket floating-point qualification.
   Reused cases remain diagnostics, not independent generalization. No training,
   extra attempts, threshold changes or promotion follows. A CPU selection
   preflight caught global `pair_index` being mistaken for a family-local index;
   selection now binds the eight explicit family/world IDs before any sampling.
