# E97 Open-SWE source-native dataset protocol v1

Status: rebuilt dataset encoding and source/mask validation implemented.
**Not yet an execution-qualified serving/runtime profile.** No deployed Pi grammar changes.
The [native episode boundary](validation/e97-native-runtime-protocol-v1.md) now
passes strict parsing/routing tests and real-source training-prefix parity;
actual shell/editor execution qualification is still pending.

Profile: `e97-open-swe-source-native-v1`.

## Record and turn grammar

A record starts with `Protocol:\n` and a canonical JSON object containing the
profile, framing instruction, and the four original source tool declarations.
Each source message follows under a blank-line-separated role header. System,
user, and tool bodies are canonical JSON objects preserving the complete source
message. Embedded newlines and control characters are JSON escaped, so message
content cannot create a structural role boundary.

Assistant bodies have exactly five lines:

```text
Analysis: <JSON string or null>
Commentary: <JSON string or null>
Think: <JSON boolean or null>
Action: <execute_bash | str_replace_editor | think | finish>
Arguments: <JSON object>
```

Analysis preserves `reasoning_content`. Commentary preserves source `content`.
Think preserves a source annotation flag; it is neither a tool call nor execution
permission. The Action/Arguments pair preserves the source function name and
all decoded argument values. Original argument-JSON spelling and call transport
IDs remain in `source_messages.jsonl`, not generated targets. No reference-patch
or model-patch oracle metadata is included.

Source reasoning and think-tool thoughts are private. Commentary and
`finish.message` remain public. A future runtime must enforce this distinction,
not expose thought text through ordinary tool UI or silently drop public content.

## Semantics and boundaries

No tool mapping is performed. In particular, absent/open-ended source view ranges
remain absent/open-ended; persistent-shell stdin and timeout arguments are not
translated into fresh bounded Pi operations. Original source observations and
errors remain attached to their original calls. Keeping these records does not
claim that the current Pi executor can execute them correctly.

One complete trajectory occupies one record of at most 65,536 p50k_base tokens.
Private reasoning plus think thoughts in a source assistant turn must fit 2,048
raw tokens and 65,536 UTF-8 bytes. Oversize or structurally unsupported trajectories
are excluded whole with an identity and reason, never truncated or segmented.
All original messages, including a terminal tool acknowledgement if present, are
retained. A final source finish call with an explicit message is required.

Assistant bodies alone receive loss. Role headers, context, and the terminal
record separator are unscored. Full-stream BPE reconstruction and exact target
boundaries are checked; a token crossing a mask boundary excludes the trajectory.
Recurrent records must reset at their beginnings; later packing must preserve
valid/reset/loss controls and cannot concatenate these records as one history.

## Split and publication

Splits group normalized repository plus original instance ID. All original
validation problems are reserved; remaining groups receive a deterministic 5%
validation assignment. No problem crosses the rebuilt splits. This is development
data with earlier-lineage exposure, **not an independent final holdout**.

Publication is owner-only, read-only, atomic, and no-replace, after full source,
mask, index, split, and round-trip validation. The manifest remains
`training_eligible:false` until runtime/source/training admission is explicit.

See [the completed rebuild report](validation/e97-open-swe-source-native-rebuild-v1.md)
for the actual inventory and cryptographic identities. The earlier 86.3M-target
sizing proposal is not the rebuilt inventory.
