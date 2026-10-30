# X42 part 3b production relaunch

The 15,822.358 tok/s historical 150M/B64 receipt is T512, bare. The matching historical T2048/B64 receipt is 7,479.019 tok/s with checkpointing (x40 receipts/part8/surface/150M_pure_B64_T{512,2048}.json). Neither is identical to this variable-head proxy.

Full-budget production smoke: 20709376 tokens / 2709.237618s = 7643.987 tok/s; the 15k-class expectation FAILED. No replacement of the measured rate or exclusion of startup steps.

Cached uninstrumented B64/T2048 engineering rerun on card5: 13288.367 tok/s across 13 steps; synchronized component probe on card5: 12208.168 tok/s. Both are engineering-only, 120s requested, two eval windows.

Component probe: loss forward 22.860s, backward (including checkpoint replay) 105.692s, AdamW 0.115s. Head-scan time 24.541s is nested in forward/backward and must not be added again.

First 13 steps of full smoke took 504.206s vs cached probe 128.228s. Cache/startup and card differences are confounded; this is evidence of early-run overhead, not proof of its sole cause.

Proceed with the explicitly requested generation after decomposition; retain lower measured throughput and all scores/degeneracy flags. Do not attest 15k throughput.

Smoke score: 6.0823864822320886; NLL: 6.080204745133718; occupancy: 0.21817370983707307; mean gate: 0.2807660283287987; mean event density: 0.13660303751627603. Degeneracy flags: {'dead_store': False, 'pressure_runaway': False, 'slope_one': False}.

Per-head telemetry and exact score components are in smoke.json. The original full-budget smoke/regime receipts are retained unchanged; successor regime changes orchestration/tests only, verified before reuse. The pre-decomposition parent refusal is retained as failure evidence. No resilient DiLoCo requirement IDs apply.

Residual risks:
- Full-budget smoke is below the requested 15k class; short cached probes do not certify full-budget performance.
- Exact cold-cache versus card/sequence-dependent slowdown remains unisolated.
- Generation 1 will be running at handoff; no winner or completed-generation claim.

Generation 1 relaunched under parent PID 1599083, population 12, preserving the original generation population. Children hold physical cards 1/5/6. Depth probe PID 634691 remains alive and untouched; cards 0/2/3/4/7 excluded. See generation_launch.json for process identities, initial telemetry, and residual risks. All 75 X40/X42 tests passed after final code changes.
