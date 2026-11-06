# Phase 3 manual execution notes

Read Phase2 result and its resume/registration instructions first. The first
unqualified root-relative lookup failed (file is under x43); no result was
assumed from that failed command. The authoritative Phase2 source registration
check passed before any GPU experiment. No Phase2 or production source changed.

Intercom/contact_supervisor transport was searched by tool name and description
at task start and was absent. No supervisor communication or approval is claimed.
The explicit work order authorizes the sustained receipt, curve, projection and
family DRAFT, and explicitly prohibits search launch. No search was launched.

Single attended implementation; no subagents, WG service, dispatcher or workers.
Only /home/erikg/emender/.venv/bin/python. No Frontier/Slurm or resilient DiLoCo
behavior; normative DiLoCo requirement IDs inapplicable. AGENTS.md/CLAUDE.md
unchanged. No live union service, sxgc or /mnt/nvme2n1 operations. Cards1/2/7
untouched. Broker allowlist3/5/6; the detached internal wrapper acquired3.

Sustained training starts from the original immutable unbounded trained model
and a fresh AdamW optimizer, exactly as the Phase2 certification slice does.
It is real sequential training on fresh verified batches, not fixed-batch timing
or an optimizer-restored microbenchmark. The resulting model is diagnostic and
is never promoted. Original checkpoint hash is checked before and after.

The active clock includes initial compilation/graph capture, shard loading and
hash verification, native cache decoding, document hashing, masked assembly,
transfer, consultation dropout, forward, telemetry extraction/aggregation,
backward/replay agreement check, clip, AdamW and synchronized completion.
Only receipt serialization/writes and atomic diagnostic checkpoints are excluded
after loop start; setup/preflight is outside the active clock and inside wall.
The first20 real updates are reported separately as warmup, not discarded from
training. Report full active, postwarm active, and wall incl. setup/exclusions.
Retain the entire per-update occupancy/event trace losslessly as gzip.

Detached wrapper Popen(start_new_session=True), DEVNULL stdin and retained
stdout/stderr. Initial survival receipt records PPID1, lease owner/starttime,
renewed heartbeat and advancing steps. Atomic checkpoints every500 steps and
at completion outside the active clock, under /home/erikg/c14_dense_bytes;
no .pt file is committed. Checkpoint recovery is not claimed as tested here;
the Phase2 paired gate separately tested its optimizer/RNG roundtrip.

The registered C14 shape is batch16/context128, key64/value128, two layers,
two heads and cap257, ops on. It is a roughly2.13M trainable fixture, not150M.
The family draft preserves velocity score arithmetic and applicable ranges,
but explicitly records raw-byte corpus and normalization differences. Historical
final_pure supplies a terminal budget class only; current velocity nominations
leave terminal duration and corpus registration pending. No existing historical
1.3B throughput number certifies the new unbounded consultation kernel.

The diagnostic starts at training-window cursor0 with seed3401 and a fresh
optimizer; it does not resume the historical optimizer/cursor. Consequently
its sequential prefix may overlap historical training exposure. This is a
throughput/telemetry receipt, not a new held-out quality verdict.

The optional physical-extent probe completed B1/T257/BS512 and B1/T512/BS1024
repeat checks and native timings. At T1024/BS2048 its first invocation entered
large LLVM/PTX compilation; the retained ptxas snapshot shows232 elapsed seconds
and4501292KiB RSS. Continued compilation exceeded300 CPU seconds and5GiB RSS.
The task agent deliberately stopped only its owned ptxas process to bound this
optional exploration. The probe catches PTXASError(-15), records T1024 failed,
and exits1; wrapper releases the lease. This is an imposed exploration stop,
not evidence that compilation is impossible or that the kernel is incorrect.
No T1024 timing, correctness or spill count is claimed. Its input PTX and
exact process/abort evidence are retained. This failure is not relabeled green.
The projection uses an explicitly conditional4x T/BS extrapolation from measured
T512; full-scale affordability remains unqualified.

Core run is complete; no training remains to resume. Diagnostic atomic model:
/home/erikg/c14_dense_bytes/emender_phase3_sustained.pt, SHA256
 d70b924097522f5de75ec0e56d1ca32975b368abbe564bf02a33634883ca8720,
step4833,9781410 targets,next_window77328,600.038887463 active seconds.
Final CPU hash/load/counter/RNG-field check passed. A future continuation needs
a newly registered runner restoring all stored optimizer/RNG/cursor fields;
current runner deliberately has no resume CLI. Do not invent a resumed-update
parity claim. For exact fresh reproduction use the original immutable trained
checkpoint056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06,
seed3401,cursor0,fresh optimizer and new output/checkpoint paths; existing
receipts and the diagnostic checkpoint must not be overwritten.

Exact next investigation: decompress extent_BS2048_compiler_input.ptx.gz to a
new temporary path (raw SHA256 in extent_compile_snapshot.json), then use
/home/erikg/emender/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
-lineinfo --fmad=false -v --gpu-name=sm_89 INPUT.ptx -o NEW_OUTPUT.cubin,
retaining wall time/RSS/stdout/stderr. This offline compiler step needs no GPU.
A successful compile still needs a fresh broker-leased native benchmark and
long-context reference/gradient qualification. Do not rerun the hardcoded
extent receipt path in place: it refuses overwrite. Do not resume/search from
the draft registration. Next engineering target is BS1024/2048 register/local
memory pressure and the one-CTA-per-sequence grid, then measured full proxy
and terminal-width training with the actual raw-byte consultation pipeline.
