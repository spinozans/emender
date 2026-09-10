# OpenHands native execution qualification v2

Date: 2026-09-10. **Passed: 18 checks, 21 real upstream tool calls.**
No model was loaded or trained. This is a bounded scripted execution gate, not
proof of model acquisition or reconstruction of every historical environment.

## What actually ran

The adapter uses OpenHands' original `response_to_actions`, `ActionExecutor`,
`BashSession`, editor, and `ConversationMemory` observation formatter. It does not
reimplement a fresh-shell approximation, translate editor views into bash, or
fabricate tool results. Native episode framing receives the real observations.

The tool definitions in all **5,969 retained source trajectories** were checked
for agreement; all four definitions matched the pinned upstream implementation
exactly, including descriptions and parameter schemas. This does not identify
the historical dataset's precise runtime version.

The successful checks covered:

- Full-file view of all 500 fixture lines, open-ended read through EOF, and finite
  inclusive ranges. A subsequent edit used a value read beyond line 200.
- Directory views remaining editor reads, and authentic invalid-range errors.
- Exact create payloads containing leading whitespace, blank lines, Unicode and
  trailing newlines; an existing file was not overwritten by another create.
- Replacement, persistent undo history, insertion and ambiguity errors without
  unintended mutation.
- Working directory and environment persistence across shell calls.
- The original ten-second soft timeout, refusal of a fresh command while another
  command was active, and `is_input:"true"` reaching that existing process.
- Acceptance of an explicit 3,600-second timeout on an immediately completing
  command; timeout/interruption followed by recovery in the same shell.
- Separate public final text and proper episode termination.

All fixture commands were authored in the qualifier. **No commands from dataset
trajectories were executed.** Tiktoken's pattern/ranks were transferred as a
hash-bound artifact and its vocabulary identity checked, avoiding a tokenizer
network download inside the sandbox.

## Isolation and lifetime

The container was created with an immutable image ID, then inspected before
startup: nonroot UID/GID 1000, read-only root filesystem, network `none`, zero
capabilities, no-new-privileges, runc, 4 GiB memory/swap ceiling, two CPUs and
128 PIDs. Only three bounded tmpfs work areas were writable. There were no host
mounts, published ports, Docker socket or GPU devices. In-container checks also
confirmed these isolation properties. Code and public tool schemas arrived via
a hash-checked, size-bounded stdin bundle, not a host workspace mount.

The runner verified terminal exit/OOM state and removed only its nonce-labelled
container, retaining inspection, raw stdout/stderr, all calls and cleanup evidence.

## Failed v1 and exact repair

- Image build v1 (`proc_5f80`) completed in 147 seconds, but execution v1
  (`proc_eb2c`) failed in three seconds before any fixture call:
  `ModuleNotFoundError: No module named 'openhands.runtime'`.
- The source acquisition correctly used owner-only permissions. Docker COPY
  changed ownership to root while retaining those modes, making the source
  inaccessible to the nonroot runtime user.
- The Dockerfile now grants read/traverse permission to this **public upstream
  code** and tests actual runtime imports after switching to UID 1000.
  Build v2 (`proc_2cb3`) passed in 11 seconds with a new image tag/ID.
- Execution v2 (`proc_20fa`) passed in **22 seconds**. The fixture assertions and
  tool-schema equality requirement were unchanged. The host runner additionally
  rejects disagreement between the recipe's image identity and the launch input.

All failed roots and the original image remain retained. No threshold was relaxed
and no failed attempt was overwritten or automatically retried.

## Evidence

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-runtime-execution-v2`

- Image: `sha256:036df5d48772c58190d5e4702976d7a865f5963358d503a35a36ec00a54726f7`
- Image manifest: `eaab33b7d60b2989e05344e60d092990f04f3f4547f521d9e343b3361c38a98d`
- Frozen inventory: `b6039d8fb1dd223ef07df348d506e55f26a0f71af46e75762c9ab662a5eb43c2`
- Recipe: `58b0a10c516cbab7f10c7b79f5804f8c2deb1b36b281aad2c9611d0ea4b5dd8a`
- Summary: `46584aa4538682e4575ad7d78905b3390392f2023aa5aaf289d400db7ecb3f59`
- Cleanup: `82799c5b6e32aa960daa9a38376218a9c8f758dc09bdb7ebe218f2772d211bf5`

The source is OpenHands commit `9ee704a25a331d0d2eb9a8e87a4dcff1d948855b`,
with upstream-locked dependencies and `openhands-aci==0.3.1`. The MIT upstream
license is retained in the image; dataset licensing/admission is a separate matter.

## Remaining work

`training_eligible:false` is still intentional. The backend has passed this
bounded fixture gate; serving/generation integration, source admission, exact
full-model optimizer/chunk qualification, sampling/mix construction and the
frozen learning experiment are not yet complete. No deployed Pi interface changed.
