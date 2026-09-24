# E97 bounded private-analysis protocol

**Status:** protocol/runtime foundations implemented; CPU protocol, trusted-controller,
and installed-Pi eight-turn round-trip gates pass. Sustained training remains blocked
on matched authority validation, recipe freeze, and fused-CUDA numerical qualification

**Protocol name:** `e97-pi-agent-analysis-v1`

## Motivation and evidence

The verified Open-SWE Pi-v2 source contains 760,526 reasoning-bearing assistant
turns from 14,363 resolved trajectories. The current converter discarded all
reasoning and supervised only actions/finals. Aggregate audit receipts:

- character audit SHA-256:
  `6bf0115d1f4337892ae828b1f69dd2f99d0ddac722c06557b0c3a4a00399c750`;
- token audit SHA-256:
  `2a93c05fe0c879c8953e0e28388a5ec6d2ff697965778a4954a86a750f9ac3fd`.

The source contains 80,894,626 p50k reasoning tokens. Median reasoning is 28
tokens per assistant turn, p90 is 330, p99 is 769, and the maximum is 8,891.
A resolved trajectory contains a median 4,713 reasoning tokens, p99 18,697, and
maximum 61,942. Naive textual delimiters are unsafe: source reasoning contains
`Action:`, `Arguments:`, and `Final:` strings.

This protocol adds a separately typed private-analysis field while retaining the
existing one-action-per-turn dispatch boundary. It is a new protocol version;
it must not silently change the deployed Pi-v2 grammar.

## Canonical native grammar

A reasoning-bearing tool turn is:

```text
Analysis: <one canonical JSON string>
Action: <tool-name>
Arguments: <one canonical JSON object>
```

A reasoning-bearing final turn is:

```text
Analysis: <one canonical JSON string>
Final: <answer>
```

The JSON string encoding makes embedded newlines and strings such as `Action:`
unambiguous and reversible. The tool name and arguments remain separately parsed
and validated. Analysis text is never interpreted as an action and is never
passed to a shell or validator.

Legacy turns beginning directly with `Action:` or `Final:` remain valid only for
legacy protocol identities. An analysis-enabled runtime must bind the exact new
system prompt and protocol digest.

## OpenAI message mapping

For a tool turn, the response uses:

```json
{
  "role": "assistant",
  "content": null,
  "reasoning_content": "bounded private rationale",
  "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": "{...}"}}]
}
```

For a final turn, `content` contains only the final answer and
`reasoning_content` contains the private rationale. The client/controller must
round-trip `reasoning_content` byte-for-byte in the next request. If Pi or any
other client drops, rewrites, displays, or fails to return the field, this
protocol is unsupported for that client and must fail closed rather than fall
back to a different transcript.

Private analysis is excluded from user-visible final content and tool arguments.
Raw traces remain sensitive artifacts under the same access controls as E97
recurrent state.

## Source conversion

- Preserve reasoning, action, tool observation/error, and final logical units.
- Target the private analysis, action, and final assistant bytes.
- Keep system, user, and tool observations as zero-loss context.
- Do not supervise failed student actions when constructing correction records.
- Never truncate reasoning silently.
- Reject or separately summarize an oversize logical unit with an explicit
  receipt; summarization requires deterministic teacher/source identity and
  semantic validation.
- Segment at complete assistant-action/tool-observation boundaries.
- Reset every recurrent layer between independent trajectories.
- Mask cross-trajectory targets and keep padding recurrently inert.

The source-admission envelope is at most 2,048 `p50k_base` reasoning tokens and
65,536 UTF-8 bytes per emitted assistant analysis. Analysis-bearing evaluation
services and sealed controllers must declare at least 4,096 completion tokens;
the legacy 512-token service limit is not an admissible evaluation of the full
source envelope. Exactly 64 of 760,526 individually audited source messages
exceed the token limit. They occur in 60 of the 14,314 legacy-action-normalizable
trajectories. The full strict converter excludes 70 trajectories at the emitted
analysis cap because source-local `think` reasoning is folded into the next
executable action before applying the bound. The immutable audit artifact SHA-256 is
`5b536a4ec4a2225ace2c75448acecc38417e5e90c45421c4fae8f577ae8c278e`;
see `docs/validation/e97-open-swe-private-analysis-cap-audit.md`. The 64K
packed-record limit remains independent and fail closed.

## Runtime requirements

The server and controller must:

1. expose an explicit analysis-protocol flag and identity;
2. stop generation only after a complete analysis plus action/final unit;
3. parse the canonical JSON analysis string before parsing the action/final;
4. enforce a completion-token and decoded-byte limit on analysis;
5. validate exactly one registered tool call;
6. dispatch only the parsed tool name and JSON arguments;
7. return private analysis in `reasoning_content` and never merge it into tool
   arguments or visible final content;
8. require the next request to contain the exact prior reasoning field when
   transcript replay is used;
9. bind the server-emitted assistant message digest and reject client-side loss
   or mutation before accepting the turn;
10. include reasoning bytes in recurrent-cache prefix identity;
11. preserve raw reasoning only in access-controlled diagnostic traces.

The trusted external controller remains responsible for authentic observations,
no-progress recovery, deadlines, and outcome receipts. The model's analysis is
untrusted text.

## Runtime gate evidence

The CPU-only installed-Pi probe completed eight streamed analysis/action/tool-result
round trips plus a final turn with nine recurrent HTTP requests and eight cache
hits after the initial miss. It retained every synthetic reasoning field exactly.
The receipt is
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi_private_analysis_roundtrip_probe_v1.json`
with SHA-256
`d8c56c26899819482d7cc664bc006e13548176dd0886ad2d09c9a151b8afb4e1`.
This proves protocol transport, not useful model inference.

Passing CPU requirements:

- legacy protocol behavior is byte-identical when analysis mode is disabled;
- JSON-string analysis round-trips arbitrary Unicode, newlines, quotes,
  backslashes, and embedded protocol markers;
- malformed, non-string, oversized, or incomplete analysis fails closed;
- one generated native turn maps to OpenAI fields and serializes back to exactly
  the same native bytes;
- streaming and non-streaming responses agree;
- Pi and the dedicated controller preserve `reasoning_content` across at least
  eight tool turns;
- token-delta recurrence and full transcript replay agree within the declared
  fused-CUDA numerical contract;
- analysis cannot select an unregistered tool or alter parsed arguments;
- dropped analysis by a client is detected before cache continuation;
- packed training preserves complete logical units, reset isolation, gradient
  isolation, and padding invariance.

All protocol-transport items now pass targeted CPU tests and the installed-Pi
probe. Fused-CUDA token-delta numerical qualification and final packed-authority
validation remain open.

## Training gate

Candidate, permanently non-promotable matched representation authorities may be
built now that the runtime round trip passes. Do not start sustained GPU training
until both authorities validate and the mixture, target-token matching, learning-rate
schedule, milestones, regression floors, and stop rules are sealed. Compare, at
matched unique trajectories and assistant-target tokens:

1. the existing action-only representation;
2. bounded private analysis plus actions;
3. optionally concise verified rationale plus actions.

Select the representation on unseen executable tasks, recovery, cycle rate,
long-horizon completion, broad retention, and token cost—not teacher-forced loss
alone.
