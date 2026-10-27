# X36 part 1 — partial: dominant replay bandwidth removed; targets remain open

This is an engineering partial delivery. The LM cost problem is **not closed**.
The baseline decomposition preceded all fixes (DECOMPOSITION.md and raw JSON).
At B64/T512/H4/S2944, replay accounts for 13,256.6 of 13,489.5 ms/step.
X34's B8 historical 23.424x ratio reproduces as 623.17 ms/step in this session.
X31's forward-only T16 measurements do not predict training replay cost.

The largest measured fix masks dead-row GKeys/GValues adjoint loads and stores
inside the existing device-loop reverse kernel, retaining the original full-S
reduction tree. B64 replay falls to 407.94 ms (32.5x faster), and complete
instrumented step to 628.51 ms (21.5x faster). B8 drops to 328.26 ms (1.90x).
Training telemetry is eval-only; its ~2.77/17.28 ms at B8/B64 disappears.
These isolated step receipts retain five raw samples after two warmups. Stream
events add no boundary synchronizations; total wall includes optimizer and
explicit final measurement sync. Separate profiler runs are excluded from
wall performance. Negative small journal ablation estimates after fixes are
retained as noise, never clamped or described as a speedup in capture itself.

An earlier reachable-prefix reverse reduction passed the original tolerance
suite but failed exact gradient comparisons in three fixtures. It was reverted
before timing. Its patch/log/XML are preserved. Final validation runs all 109
unchanged X31 oracle tests **through X36**, plus four exact output/gradient/route
comparisons and one exact train/eval telemetry comparison: **114 passed**.
Existing imported-state, continuation, repeated-backward, rollback and mutation
checks remain green. No historical X31/X34/X35 source is modified.

Full-step profiler still records 1536 token kernels (three per token). The
reverse kernel already loops over all tokens on-device: it was not a
per-token-launch backward. CPU launch API totals in optimized runs are
5.31/7.34 ms at B8/B64; these are overlapping CPU API times, not an additive
estimate of device launch gaps. Journal increment is a matched no-grad forward
ablation, not isolated in-kernel journal timing. Attribution is therefore
explicitly partial. The steady loop still has one clip-norm DtoH/stream sync,
beyond measurement synchronization. Removing telemetry eliminates twelve
stream synchronizations and twelve DtoH activities in the B8 profile, but
**global zero host syncs FAILS**. The 175 local-scalar events include CPU
optimizer scalar handling and should not all be called GPU synchronizations.

The persistent forward, <=10 launches, T4096 <=2x, and complete T/S asymptotic
validation are not delivered. WORK_ORDERS.md names the remaining work. This
slice does not authorize resuming scientific LM pilots/reports.

Operational evidence: broker-leased physical GPU0 on lambda01; exact project
Python. Certified 2/3/4 and forbidden 7 excluded by hardcoded wrapper whitelist.
No .pt output, WG, delegation, Frontier/Slurm or resilient training changes.
Partition/QOS and resilient requirement IDs are inapplicable. Callable catalog
search found no contact_supervisor tool, so no Intercom coordination was possible.
Unrelated workspace changes were preserved. The requested summary_console.txt
was absent; X34's smoke_summary.json was used as the historical summary.

Matched X34-style end-to-end rerun completes **2,097,152 tokens / 512 steps per
arm** without checkpoints. Optimized tape median 332.522 ms, base 23.260 ms:
**14.296x**, versus the historical 23.424x. Both curves match all 512 smoke
steps with max absolute loss error **9.5367e-7** (registered check <=1e-5).
They are not bit-identical cross-run loss receipts; even the unchanged base
shows this same sub-micro loss difference. Cell forward/gradient exactness is
checked separately against X31 in-process. Matched cost/semantics evidence is
in receipts/training_slice.json; no learning-quality improvement is claimed.

Matched B64/T512 comparison: base **167.624 ms**, tape S2944 **640.866 ms**, ratio **3.823x: FAIL** (<=3).

| S | Complete tape step ms | ms/token |
|---|---:|---:|
| 736 | 350.275 | 0.684 |
| 2944 | 640.866 | 1.252 |
| 11776 | 4071.169 | 7.952 |

This fixed-T capacity sweep does **not** establish the requested asymptotic
claim. The S2944->11776 increase is ~6.35x for 4x S, with full-S replay
reductions and dense state retained. No T sweep was run. T4096 gate remains
unmeasured; no passing extrapolation is made. All raw samples are retained.

Final validation strengthens the four exact fixtures to compare raw tensor
bytes (including zero sign bits), and exercises keyword append_err_coef API
compatibility: **114 passed in 13.49 s**, final_api_parity.txt/.xml. No kernel
or arithmetic changes followed performance collection. sources.json hashes
the final implementation, tests and historical source references.
