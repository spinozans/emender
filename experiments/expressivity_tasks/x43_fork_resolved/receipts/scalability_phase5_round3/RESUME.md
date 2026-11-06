# Round3 exact resume — rejected split candidate, no training

Decision NO-GO. Do not install this candidate in production or start the1.3B
scenario from its speed measurement. The matched round2 selected kernel remains
faster and itself unqualified. No new training/checkpoint exists for round3.

Committed pre-implementation profile/design84aaeba0; implementation9c98ff35.
Final result is RESULT_UNBOUNDED_SCALABILITY_PHASE5_ROUND3.md. Authoritative
index validation_summary.json binds source bytes and terminal exits. axis_profile
is the invalid clamped first attempt; axis_profile_fixed isolates slot and time
axes. quick is the retained compile failure; quick_fixed passes. sweep is the
ten independent monolithic pairs. matched_r2 compares against the previous
selected candidate. compile2048 is candidate-only and must never replace the
failed independent long_2048 gate.

Completed: ten native/full-state/raw-trace repeat pairs,109 X31,108 operator,
eight short state and four replay/gradient pairs, T512/T1024 masked reset/full
fingerprints,24 primitive adversarial byte pairs. T2048 unchanged reference
still exits124 at180s, before candidate or replay. Candidate at S4097/BS4096
executes after a14.140219s first compile+call and repeats its own fingerprints;
this is not independent reference/state/replay/gradient acceptance.

Split candidate sources are c14_phase5r3_split_kernel.py and opt-in validation/
probe modules. READ uses256-slot launches and carry[B,2,128] FP32. Score uses
128-slot CTAs. Control/normalization still carry BS-wide scalar metadata; they
do not meet a stronger O(1) total-state/compile-complexity guarantee. At matched
T1024 the candidate is368.719367ms/head versus round2's279.310420; projected
733.491217native-only days. The earlier independent-reference sweep's446.030798ms
is retained. No occupancy selection or eventual speedup is promised.

Next structural work, if pursued: preserve softmax's exact BS reduction tree
while splitting scalar control (tree-frontier carry), and remove serialized READ/
launch overhead without regrouping additions. This needs a new pre-implementation
design receipt, new source binding, and fresh independent byte gates. Separately,
the original T2048 monolithic reference LLVM optimizer remains a certification
bottleneck; retain original S4097/BS4096 and arithmetic, do not silently weaken
or substitute the reference. Source-level/kernel splitting of the candidate
alone has not certified equivalence at T2048.

Any rerun must use a fresh receipt name/directory and a copied, explicitly
registered launcher with broker physical3/4/5/6, never protected1/2. Existing
launchers refuse completed receipts. Use only /home/erikg/emender/.venv/bin/python.
Independent long command (under the leased wrapper, fresh --output):

/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r3_long_parity --tokens 2048 --output <fresh-round-directory>/long_2048.json

The full suite wrapper, split sweep and followups are run_c14_phase5r3_*_leased.sh.
They cannot be rerun on their existing completed outputs. Retain compiler limits,
failed attempts, all terminal exits and source/index hashes. Push every commit.
The standing real1.3B GO also requires trained-model fingerprints, >=2M paired
real targets/optimizer/RNG/cursor/checkpoint resume and measured throughput/memory;
no such work has been qualified here.

Seed3402: both train terminals exit0,35419steps/71677042targets each. Budgeted
full eval terminal0/table exists. Unbudgeted evaluator PID714812 remains under
detached broker owner502751 (PPID1, starttime403545982). Two snapshots in
twins_state.json verify live process, renewed lease and advancing evaluation
windows. Do not restart or disturb this evaluator. Its full terminal/table was
absent at harvest; no second-seed sign may be inferred from partial NLL. Upon
completion harvest both66-row tables and denominators against
PREREGISTRATION_C14_B_SEED3402.md, then bind the actual paired final-test delta.
Seed3401's -0.019429161 remains single-seed evidence until then. Delivery process
already exists from the earlier session; no new service or watcher was started.

No supervisor transport was available in discovered tools. No contact claimed.
No own long training or persistent GPU process should remain; completion_check.json
records fresh own-process/lease/device checks. No DiLoCo changes; requirement IDs
inapplicable. AGENTS.md/CLAUDE.md and unrelated worktree changes remain untouched.
