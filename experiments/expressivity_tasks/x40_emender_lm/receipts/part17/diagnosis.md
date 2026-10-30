# X40 part 17: failed ordering experiment; scoped repair remains open

**FAIL / incomplete. No production repair or full recertification is claimed.**
This commit uses the work order's triage + scoped-plan fallback. The diagnostic
changes only the process-local Triton JIT source; repository production sources
remain unchanged. The requested green-completion commit title would be false.

## Hazard triage

`hazard_inventory.json` retains every access endpoint and hazard count in each
of the 35 original racecheck groups, with stable IDs R01–R35. All are treated
as REAL repair obligations, conservatively. None is dismissed as benign. The
reports include shared scratch read/write and write/write conflicts around
lane-dependent reductions. Identical writer values and private ownership have
not been proved. This is a fail-closed classification, **not** a completed
instruction-level proof of the lifetime responsible for each group or the
assertion. Source line numbers alone cannot establish shared scratch ownership.

U001–U100 enumerate every printed uninitialized access in the original isolated
report, including instruction, thread/block, address and width. All are REAL
uninitialized reads at kernels.py:102. The retained part-16 report suppressed
8,124 further accesses; the inventory explicitly marks that coverage gap.
An unlimited-print unchanged-baseline rerun removes that restriction: application
PASSED, sanitizer exit 86, **8,224 errors**, all **8,224 accesses printed**, all
at scan_kernel+0x1baf0 / kernels.py:102. `initcheck_baseline.txt.gz` losslessly
retains the raw report. `baseline_init_summary.json` individually enumerates
all 8,224 distinct width/instruction/thread/block/address records and classifies
the entire set REAL. Its decompressed SHA256 permits raw-report verification.
This completes the uninitialized-access inventory for this isolated replay;
it does not claim all possible shapes have only this one uninitialized site.

`ReadStrengthCache` is allocated with `empty_like`. Newly allocated live slots
can be read before a cache store. `tl.where` chooses *after* loading and does
not suppress that read. A candidate mask requires both live and
`r_touch < Start+t-1`: if touch is newer, the selected base is the reference,
so the cache is unnecessary. If touch is older, a previously live slot should
have been cached at the previous read. This reasoning needs continuation,
POP/reallocation, imported-state and all-shape tests before promotion.

## Bounded experiment and failure

`diagnostic.py` inserts 11 barriers around read reductions and after the history
store, and masks the unused cache load. It preserves arithmetic expressions,
reduction tree shapes, launch geometry and assertions. It is deliberately a
broad diagnostic, not a minimal correct repair. Its source hashes are printed
at launch. `run_diagnostic.sh` supplies sequential broker leases and tool logs.

At B8/T2286/S2944, replaying the captured scan-10 input:

- Racecheck **fails**, exit 86: **18 displayed error groups**, followed by the
  same int32-add overflow assertion in block 5. `racecheck_diagnostic.txt.gz`
  retains every displayed endpoint. This cannot be called zero real hazards.
- Initcheck **passes**, exit 0, **0 errors**, application prints PASSED.
  `initcheck_diagnostic.txt` is an isolated diagnostic result only. It does not
  establish production clean status, output parity, or causal independence of
  the cache defect. The mask and barriers were changed together.

The previous isolated baseline had 29 groups, whereas the full-batch baseline
had 35. Different source offsets and schedules prevent a one-to-one subtraction
of groups. No claim that a specific number of hazards was repaired is made.
Broad source-level barriers leave a generated-reduction scratch problem open;
adding barriers at these boundaries is not sufficient evidence of correctness.
No production candidate was installed, so there is no production change to
revert. The failing process-local experiment is rejected.

## Scoped next repair plan and stop gates

1. Map every R01–R35 endpoint PC to the exact captured cubin/SASS, shared byte
   range and generated reduction phase. Retain the tool/compiler versions and
   cubin hashes. Determine which accesses share scratch across iterations,
   layout conversions or debug overflow-combiner reductions. A source line
   appearing in a report is insufficient to declare identical writers.
2. Reduce the *generated* scratch-lifetime failure while retaining debug=True,
   four warps, operand values and reduction topology. Test ordering boundaries
   inside layout-conversion/reduction phases. If Triton lowering is responsible,
   retain an upstream-quality minimal reproducer and review the generated code
   for any toolchain repair. Do not switch off assertions, change integer widths,
   change warp counts or substitute another reduction tree as an accepted fix.
3. Isolate the cache mask from the barriers in a separate candidate. Prove the
   selected operand unchanged for every timestamp branch; compare all public
   values, native state and trace fields, including continuation and gradients.
   Zero-initialization is another candidate if its graph and allocation behavior
   is acceptable. Neither is a repair for the surviving shared scratch races.
4. Require fully instrumented racecheck and initcheck at the failing shape and
   a receipted certified shape, plus 100 *full production* reproductions. Capture
   tool exits and accounting; reject timeouts and filtered producer tracking.
   Retain any claimed benign group's lane/value/ownership proof individually.
5. Only after sanitizer success run the complete existing all-shape/all-policy
   X31 parity gate, with bitwise values/state/traces/pressure and gradients,
   all 109 oracle tests, graph/eager checks and the >=2M-token bit-matched curve.
   No pressure exclusion, tolerance loosening or reference substitution.
   Any gate failure rejects/reverts the candidate and gets its own receipt.
6. Run the original part-4 unchanged-X31 pressure probe, writing new output
   under part17 rather than overwriting part4. It imports the **separate X31
   kernel**, so an X40-only patch cannot mechanically repair that reference.
   A shared source pattern is not proof of one runtime cause. If an X31 repair
   becomes necessary to pass strict pressure parity, resolve the reference
   scope explicitly before promotion; keep the unchanged probe as control.
7. Only then finish the production 9x bin and budget-scaled depth evaluation.
   Before evaluating, hash-bind budgets, expected range ordering, checkpoint,
   seeds, metrics and prediction against the X28E ladder. Report failed or
   absent bins as such. No depth verdict or payoff follows from this diagnostic.

## Execution scope and remaining acceptance

Host lambda01; interpreter /home/erikg/emender/.venv/bin/python only. GPU
experiments ran sequentially through the broker restricted to 2/3/4 and selected
card 2. No card 7 selection, search interruption, agents, WG service, Slurm,
or resilient training changes. DiLoCo requirement IDs are inapplicable.
The broker's lease files reside under .wg; the explicitly requested broker is
used without starting WG orchestration. Unrelated workspace files are untouched.
No .pt is committed; the existing /tmp capture is external replay input.
The runner was extended between runs to select the tool and then the baseline.
The first racecheck run used the sanitizer default print limit; all 18 groups
were displayed. Both initcheck runs used --print-limit 0. The original generic
lease files retain the racecheck lease. The initcheck-named lease files retain
the last baseline lease (the intermediate diagnostic initcheck lease snapshot
was overwritten). All three tool calls selected physical card 2; the terminal
broker snapshot shows no remaining card-2 lease. The current runner uses unique
tool/mode receipt names to prevent another overwrite.

Full triage proof, certified-shape sanitizers, 100 production repeats, all
recertification gates, the part-4 rerun, 9x acceptance, budget prediction and
final depth verdict remain **not run / not satisfied**. No full completion is
attested. `contact_supervisor` was searched first and again by name/description;
the tool is unavailable. No substitute communication channel was used.
