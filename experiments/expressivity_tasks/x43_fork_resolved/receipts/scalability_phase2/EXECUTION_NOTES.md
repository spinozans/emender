# Execution notes

Only `/home/erikg/emender/.venv/bin/python` was used. No subagents or WG.
No live union service operation, sxgc, protected GPU workload operation,
checkpoint overwrite, training-scale launch or committed .pt artifact.
Supervisor/intercom transport was searched by name and description and is
absent; no supervisor contact or approval is claimed.

The first global-staging candidate fails full-model byte identity despite
8.15117 diagnostic compute steps/s and exact synthetic zero-value probes.
Retain this failure; it is not a qualifying speedup. Explicit-layout Gluon
restores the standard full-model fingerprint: 6.90254 diagnostic steps/s,
105 C14/operator tests and 109 X31 oracle tests pass, and all four nonzero
occupancy state/replay/gradient fingerprint pairs pass. The final row-key variant subsequently passes its own complete scoped battery,
988 paired updates / 2,000,483 targets, and atomic checkpoint/RNG roundtrips.
These are disposable certification gates; no production training is authorized.

Gluon construction attempts v1-v6 fail compilation and are retained as failed
experiments. A stray preliminary shell probe failed with IndentationError
before kernel execution. The first row-key experiment fails compilation
because Gluon has no broadcast_to; v2 uses an exact conditional broadcast.

A validation launcher acquired card5 just before the card3 Gluon profile
finished. Its Python process was explicitly paused during final profiling,
then resumed after the profiler exited. Do not describe the launchers as
having strictly disjoint lifetimes. No intentional concurrent GPU experiment
was run. The final row-key certification runs only after both earlier
launchers exit, on leased card3, with a complete source snapshot.

GPU occupancy values in the baseline are analytic bounds from grid/compiler
resources, not Nsight achieved occupancy. ncu was initially absent from PATH; a later explicit-path probe fails with ERR_NVGPUCTRPERM. Internal native
phases are fused; per-operator timing cannot isolate them. Global spill count
is compiler metadata and differs between synthetic and trained configurations.
257 live physical slots require BS512; BS256 cannot cover that state.

No resilient DiLoCo behavior changes: gap-matrix requirement IDs inapplicable.
Existing unrelated workspace changes are preserved. AGENTS.md/CLAUDE.md are
unchanged. Phase-1 registration matches committed/working source bytes before
experimentation. No existing source-bound production file is edited.

The installed ncu was discovered late. Its probe (physical card3 / visible
device0) fails with ERR_NVGPUCTRPERM and a torch.profiler/CUPTI subscriber
conflict; its timings are excluded. No privilege or driver changes attempted.
The first staged whitespace check failed on raw profiler log padding and three
extra EOF blank lines. Logs/source snapshots were compressed losslessly, and
the EOF-only normalization is AST-audited, registered and followed by the
complete 2M-target slice. The final scoped whitespace check passes.
