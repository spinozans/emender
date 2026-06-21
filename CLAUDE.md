<!-- workgraph-managed -->
# workgraph (project-specific guide)

This file is the **layer-2** project guide for agents working in this
workgraph project. It is NOT the universal chat-agent / worker-agent
contract — that is bundled inside the `wg` binary and emitted by:

```
wg agent-guide
```

Run `wg agent-guide` at session start (or read its output from a previous
session) to get the universal role contract: chat agent vs dispatcher vs worker
distinction, `## Validation` requirement, smoke-gate, cycle handling, git
hygiene, worktree isolation, "no built-in Task tool" rules, etc.

This file only covers things specific to this project. Add project-specific
build commands, test commands, architecture notes, and service recipes here.

**At the start of each session, run `wg quickstart` in your terminal to orient yourself.**
Use `wg service start` to dispatch work — do not manually claim tasks.

This guide is written to both `CLAUDE.md` and `AGENTS.md` and kept in
lock-step. The two files exist because Claude Code and Codex CLI look for
different filenames, but they should never drift in content. Any divergence is
a bug. Update both together.

## NON-NEGOTIABLE #1 — FUSED TRITON KERNELS ONLY. NO EAGER. NO "PYTHON FIRST."

**Every experiment runs the recurrence / state dynamics through the FUSED TRITON
kernel. There is NO such thing in this project as an experiment that runs the
recurrence in eager / pure-PyTorch.** Not for "quick validation," not for
"prototyping," not for a "sanity check," not "just to see the signal first."
If you catch yourself proposing an eager/Python path, STOP — that is a bug in
your reasoning, not a shortcut.

**Why eager is forbidden here (it is not a cheap preview — it is a DIFFERENT,
non-transferable experiment):**
- The fused E97 kernel was **INERT** in the autocast path while eager "worked"
  (`E97 fused LM wiring`). Eager signal that vanishes in fused.
- TF32 fused was **untrainable** while eager looked fine (`complex-eig-lm-fused`).
- Eager throughput is **meaningless** — every wall-clock / accept-reject verdict in
  this repo depends on the fused-kernel reality, on a no-NVLink DDP-bound box.
- bf16/precision and chunk-vs-eager numerics differ enough to flip conclusions
  (`complex-eig chunked overflow`, `e97-chunked-kernel`).

**The rule, concretely:**
- The recurrence/state update (the per-step or chunked scan) MUST be the fused
  `@triton.jit` kernel. `--use_triton 1`, bf16, and the `[fused-guard] ... NO
  eager fallback` assert MUST be present and pass in every run. A run without the
  fused-guard, or with ANY eager fallback, is **INVALID and scores 0 at the gate.**
- Need NEW state information (a state summary, an extra readout, a probe)? You
  implement it **inside the fused kernel + the matching backward VJP** (reverse-
  replay BPTT). If you cannot put it in the kernel, you do **not** run the
  experiment. "Validate in eager first" is not allowed.
- Standard dense layers (Linear / SwiGLU MLP / `o_proj`, i.e. cuBLAS matmuls on
  the kernel's *outputs*) are fine — they are not the recurrence. The ban is on
  running the **state dynamics** in eager, ever.

If an experiment cannot be done in a fused Triton kernel, the answer is to write
the kernel, not to fall back to Python.

## GPU leasing — ALWAYS lease before touching a GPU

This box has 8 GPUs and **no central allocator**. Multiple workgraph agents run
concurrently. If you hand-pick `CUDA_VISIBLE_DEVICES` you **will** collide with
another agent and both jobs slow down or OOM. Use the lease broker instead.

### The one-liner (put this at the start of any GPU task)

```bash
# Reserve N exclusive GPUs (waits/round-robins until N are free), exports
# CUDA_VISIBLE_DEVICES, auto-releases when this shell exits.
eval "$(scripts/gpu_lease.sh 2)"        # or: eval "$(scripts/gpu_lease.sh acquire 2)"
echo "got $CUDA_VISIBLE_DEVICES"
python train.py ...                      # runs only on the leased GPUs
# lease is released automatically on shell exit (EXIT/INT/TERM trap)
```

A GPU is granted only when it is **both** (1) physically idle per `nvidia-smi`
(util < 10% **and** mem < 256 MiB) **and** (2) not already held by a live lease.
Acquisition is atomic: the whole reserve operation runs under one `flock`, so
concurrent callers never get the same GPU. If fewer than N qualify, the call
**waits** (polling) until enough free up — that is the round-robin/queue.

Notes:
- Calling the one-liner **twice** in the same shell *accumulates* leases (both
  sets release together on exit); it does not drop the first.
- `release <id>` only frees a lease **you** own (pid+host); use
  `GPU_LEASE_FORCE=1 ... release <id>` for operator override.
- **Single host only.** Lease files are keyed by bare GPU index, so the lease
  dir must not be shared across machines.
- Exclusion is exact **between broker users**. Protection against a GPU job
  launched *outside* the broker is best-effort (it relies on the nvidia-smi
  idle snapshot). Route **all** GPU jobs through the broker for hard exclusivity.

### Subcommands

| Command | Purpose |
|---|---|
| `scripts/gpu_lease.sh acquire N [--wait\|--no-wait] [--timeout S]` | reserve N GPUs (default: wait forever) |
| `scripts/gpu_lease.sh release [id,...]` | release my GPUs (default: all leased by my PID) |
| `scripts/gpu_lease.sh heartbeat [id,...]` | refresh heartbeat (long jobs; the eval one-liner does this automatically) |
| `scripts/gpu_lease.sh status` | show physical GPU state + active leases |
| `scripts/gpu_lease.sh list-free` | print currently grantable GPU ids (csv) |
| `scripts/gpu_lease.sh reap` | reclaim stale leases now |

### How leases & reclamation work

- Lease files live at `.wg/gpu_leases/<gpu_id>` (`.wg` is a **shared** symlink,
  so leases coordinate across **all** worktrees on the box). Each records
  `pid`, `starttime` (PID-reuse guard), `host`, `created`, `heartbeat`, `adopted`.
- **Release** happens automatically via an `EXIT/INT/TERM` trap installed by the
  eval one-liner, or explicitly via `release`.
- **Stale reclamation** is automatic on every acquire: a lease is reclaimable if
  its **PID is dead** (verified by `/proc` starttime so a recycled PID can't
  resurrect it) **or** its **heartbeat is older than the TTL** (default 900s =
  15 min). The eval one-liner runs a background heartbeat keeper so live jobs
  never expire; if your process dies (crash, `kill -9`, OOM) its GPUs free up.

### Tuning knobs (env vars)

`GPU_LEASE_TTL` (default 900s) · `GPU_LEASE_THRESH_UTIL` (10) ·
`GPU_LEASE_THRESH_MEM` (256 MiB) · `GPU_LEASE_VISIBLE` (csv allowlist of GPU
ids the broker may consider) · `GPU_LEASE_DIR` · `GPU_LEASE_POLL` (retry secs
while waiting) · `GPU_LEASE_HB_INTERVAL` (heartbeat keeper interval) ·
`GPU_LEASE_ADOPT_GRACE` (secs a fresh lease is protected while the holder shell
binds it, default 10) · `GPU_LEASE_FORCE=1` (override release ownership check).

### Don't

- ❌ Don't set `CUDA_VISIBLE_DEVICES` by hand for multi-GPU/concurrent work.
- ❌ Don't delete files under `.wg/gpu_leases/` by hand — use `release`/`reap`.

### Verify

```bash
bash tests/test_gpu_lease.sh   # real concurrency + stale-reclaim suite (needs idle GPUs)
```
