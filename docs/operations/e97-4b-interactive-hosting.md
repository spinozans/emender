# E97 4B interactive hosting — operator runbook

Serve the promoted E97 4B Pi-native agent checkpoint
(`v6-u96`, SHA-256 `d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b`)
through the repo's OpenAI-compatible server and talk to it from Pi.

## Why not llama.cpp / GGUF

E97 is a hybrid recurrent architecture (tanh-state sequential recurrence,
custom Triton kernels, Emender's own model code). There is no llama.cpp
support and no GGUF conversion; writing one is a large, numerics-risky
project (documented bf16/fp32 recurrent-state precision sensitivities). The
supported path is the repo's own server, which loads the exact checkpoint
with the same code path used by every passing evaluation.

## Why the server runs from a pinned clean snapshot

The launcher materializes a read-only `git archive` snapshot of the pinned
serving commit (default: repo HEAD — its serving stack is byte-identical to
the tree that produced the passing v6-u96 dual gate) and runs the server
from there, exactly like every gate evaluation. The live checkout carries
uncommitted work (a stricter runtime-attestation layer in the serving
stack) whose integrity check deliberately rejects the dirty tree because
the working-tree controller has drifted from the sealed first-party
archive. Do not "fix" this by serving from the dirty checkout; reconcile
the attestation work with the sealed archive separately. The launcher
compensates for the snapshot's lack of a server-side `--checkpoint-sha256`
flag by verifying the checkpoint digest itself before acquiring the GPU,
and records `checkpoint.sha256` + `source-commit.txt` next to the logs.

## Preconditions

- One idle GPU. The lease broker (`scripts/gpu_lease.sh`) only grants
  physically idle GPUs, so while the wide-arc training holds all eight, the
  launcher waits safely instead of colliding. Run this when training is
  paused (juncture windows) or complete.
- The first-party controller archive must be present (server import reads
  it): `configs/pi/e97-firstparty-generator-manifest-v1.json`,
  `configs/pi/e97-firstparty-source-v1.tar`.
- Tokenizer cache: `/tmp/data-gym-cache` (the launcher sets
  `TIKTOKEN_CACHE_DIR`).

## Start the server

```bash
cd /home/erikg/emender
scripts/serve_e97_interactive.sh            # foreground supervisor; use tmux/screen
scripts/serve_e97_interactive.sh status     # is it serving?
scripts/serve_e97_interactive.sh stop       # stop server + release the GPU lease
```

`start` verifies the checkpoint SHA-256 pin, materializes the pinned clean
snapshot, acquires one GPU, launches on `127.0.0.1:8797` (the canonical
`emender-local` port), waits for `/health`, and prints the ready line.
Logs and identity records: `/tmp/e97-interactive-serve/`.

The server pins the **canonical Pi-core system prompt server-side**
(`--pi-core-canonical-system`), matching the frozen panel the checkpoint
passed. Whatever system prompt Pi sends is overridden — do not "fix" this;
the model is trained for exactly this prompt.

## Verify before Pi

```bash
curl -s http://127.0.0.1:8797/v1/models | jq .
curl -s http://127.0.0.1:8797/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"e97-dense-agent","messages":[{"role":"user","content":"List the files in the current directory."}]}' | jq '.choices[0].message'
```

Expect a `tool_calls` message (a `read`/`bash` call), not prose — the model
speaks the eleven-tool Pi surface: `read`, `bash`, `edit`, `write`,
`process`, `ffgrep`, `fffind`, `web_search`, `source_check`, `fetch_content`,
`get_search_content`. No API key is required on loopback (the server only
enforces `Authorization: Bearer` when `--api-key`/`EMENDER_AGENT_API_KEY` is
set; the Pi-side `apiKey: "local"` is a harmless placeholder).

## Talk to it in Pi

The provider `emender-local` is already registered in
`~/.pi/agent/models.json` (mirroring `configs/pi/e97-dense-agent.models.json`).
From any working directory you want the agent to operate in:

```bash
pi --provider emender-local --model e97-dense-agent \
   -e /home/erikg/emender/configs/pi/e97-core-tools.ts \
   --no-builtin-tools --no-skills --no-context-files
```

This is the exact tool surface (`e97-core-tools.ts`) and provider/model
spelling used by the frozen core panel. The declared context window is
32768 (the proven panel config); training packs were 64K, but long-context
interactive sessions are precisely the unqualified conversational-drift
regime — treat >32K sessions as exploratory.

## What to expect (honest calibration)

Promotion-gate results for this checkpoint, measured on frozen panels:

- Execution 71/96 on the 96-case real-workspace panel (fresh 32/32,
  prior-fresh 16/16, prior-regression 11/16, transfer 8/16,
  **composition 4/16**).
- Pi-native tool panel: 12/14 valid frames, 10/14 correct first actions.
- All retention windows passed.

Known weaknesses to watch for while testing:

1. **Multi-step composition is the weakest axis** (4/16): tasks needing
   several dependent edits in sequence will often derail mid-chain.
2. **Conversational drift on long chats**: quality decays over long
   sessions; start fresh sessions for new tasks.
3. **One marginal Stage-B case flips valid/invalid between runs** — a
   single odd tool frame is inside the measured noise band, not a regression.
4. Error recovery after a failed tool call is much better than pre-v6 but
   not perfect; loop/stall behavior on hard failures is the thing the
   program is actively training out.

Failure transcripts (especially loops, stalls, malformed tool calls) are
directly useful — they feed the program's detector-driven data generation.

## Stopping

`scripts/serve_e97_interactive.sh stop` (or Ctrl-C in the supervisor
terminal). The EXIT trap kills the server, releases the GPU lease, and
clears `GPU_LEASE_HELD`. Verify with `scripts/gpu_lease.sh status` that no
lease is left held.
