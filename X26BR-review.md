## Review

**Verdict: ACCEPT-WITH-CONDITIONS** — acceptance of the machinery, not training outcomes.

Paths below abbreviate:
- **N** = `experiments/expressivity_tasks/results_stage26b_native_control`
- **R** = `experiments/expressivity_tasks/results_stage26_replication`

### Correct

1. **Resource censor is supported and is not a training verdict.**
   - `R/rnn_graph_seed0/training.jsonl:134–153` confirms update153 at59,890.7566s. The observed interval gives **12.3549 updates/hour**, matching `N/receipts/graph_eager_censor.json:6–18`.
   - That receipt retains CPU100% and GPU2 utilization10%; `R/receipts/rnn_graph_seed0/console.txt:1` independently records update-zero validation taking13,898.2763s.
   - `R/markers/rnn_graph_seed0.CENSORED:1–8` matches the terminal receipt: censored, exit143. `R/rnn_graph_seed0/result.json:3–5` explicitly says **“not adjudicated; right-censored.”**
   - Supervisor independently attested authorization, referencing `a85bc7bf-1d44-4f9b-8e4e-bcfb01eddd8f`.

2. **Native control deliberately preserves the five contract differences.**
   - `R/sources/LinearRNN/imm_unbound/gen.py:26–40,65–75,127–160` implements int64 multiplication, `==0` labels, uniform entries, unbalanced default and per-length sampling.
   - `N/COMPARISON.md:7–27` explicitly compares all five against X23E. Checked against `experiments/expressivity_tasks/tasks/merrill26_separators.py:52–60,87–88,108–115,149–160`: arbitrary-precision products, `>0`, probabilities(.45,.10,.45), scalar9N tokens and global balancing.
   - Native BOS+matrix-vector inputs, intermediate MAT auxiliary targets and no STATE teacher forcing are supported by upstream `train_rnn.py:229–295` and the control runner’s `forward()`.
   - Reconciled BCE/AdamW settings and eligible-pool curriculum are implemented; the comparison explicitly discloses the scheduling differences from native bucket/partial batches. This is **not a tokenization-only intervention**.

3. **Predictions precede launch.**
   - `N/REGISTRATION.md:19–28` and `N/registration.json:16–20` freeze success→contract-bundle divergence, failure→protocol reconciliation implicated/raw defaults next, and a mixed-outcome branch.
   - They prohibit single-field attribution and majority-accuracy-only success claims.

4. **Failed fused gate correctly prevented relaunch.**
   - `N/receipts/scan_tests.txt:11–20` records **0.2734375 >0.08** BF16 saturation input-gradient error: one failure, two passes.
   - `N/receipts/graph_scan_deferred.json:2–15,22–40` explicitly rejects launch, records zero optimizer updates and retains the eager censor/dead-process evidence.
   - Ordinary-case passes are not presented as overall qualification. No production Python reference to `merrill26_clamped_scan` was found in the named implementation seam.

5. **Tests, smokes, launch and watcher receipts are present.**
   - `N/receipts/tests.txt:6–13`: **38 passed**.
   - Both `smoke_*_integer.json:1–13`: passed, N300/batch128, zero updates; runner smoke performs backward, finite-gradient clipping and evaluation.
   - `N/registration.json:4–5`, `receipts/launch.json:2–5` and supplied history establish registration/push before launch.
   - `receipts/launch_verified.json` records **GRU card4/PID1369677**, **DeltaNet card5/PID1369696**, identities, UUID bindings and configs.
   - `scripts/watch_merrill26_native_control.py:40–97` and `N/receipts/watcher_wiring.json:2–17` provide identity-safe resource guards,60s polling and supervisor probe wiring. Native notification arming is correctly **not claimed**.

**No issues found.**

### Exact acceptance conditions

1. Retain the supervisor’s committed-content and authorization attestations with this review. I inspected current source/receipts plus supplied history/stat artifacts—not an independently obtained committed patch.
2. Parent completes checksum/provenance audits and reruns the documented38-test command in `N/VALIDATION.md:7–10`; this reviewer executed no commands or GPU workloads.
3. Graph censor remains standing: **no fused relaunch until the complete forward/gradient equivalence gate passes**. Raw-script-default controls require separate registration; native outcomes remain unadjudicated.

**Merge verdict: OK with notes**, subject to those conditions.

Persistence: complete artifact returned for parent/runtime persistence; no write-capable tool was available.
---

## Parent attestation (2026-10-24)

X26BR conditions satisfied: (1) The supervisor's committed-content and authorization
attestations are retained with this review (the full authorization transcript is in the
supervisor reply receipt 62f68168-0a77-4a4a-8abe-a8a962449843: the eager resource censor
at the measured 12.35/h rate as INFRA-RESOURCE-INFEASIBLE not a training verdict; the fused
relaunch ONLY behind the forward/gradient equivalence gate with fail-closed deferral; the
native control as unmodified shipped code with all five contract differences deliberately
preserved under the reconciled protocol, with the pre-registered decision tree). (2) Parent
audits complete: the documented 38-test command (VALIDATION.md scope: native control +
replication + separators + recipe-controlled) reruns green under the parent - 38 passed in
7.32s; upstream snapshot identity re-verified - all 14 vendored LinearRNN files hash-match
the sources/manifest.json at dbfa365a; local HEAD equals origin/main through ab1d71a1.
(3) The graph censor STANDS: no fused relaunch until the complete forward/gradient
equivalence gate passes; raw-script-default controls require separate registration;
native outcomes remain unadjudicated pending their terminal gates.

Verdict: X26BR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X26b native-control machinery closed.
