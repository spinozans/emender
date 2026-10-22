## Review

**Acceptance verdict: ACCEPT-WITH-CONDITIONS.** The measurements are supported; item (6)’s position-local interpretation is not.

### Correct

1. **Preregistration preceded evaluation.** Parent Git audit attests `eb7f90e7` at 18:01:09Z → source/pool freeze `42ceb5ed` at 18:07:11Z → results commit `9dc67f28` at 18:20:04Z. Independently audited receipts place the earliest evaluation batch at **18:07:27Z**, after both registration and freeze. Committed history was parent-attested, not directly accessible through my review tools.

2. **Ladder accounting and tables agree.** CPU audits verified **216 rungs, 3,456 evaluation batches, 96 control batches, zero censoring and zero updates**. All reported ≥480/512 survivor-count profiles match receipts. Native profiles are:
   - T16 E97/GDN2: four throughout.
   - T64 E97 cold/curriculum/wean: two/four/two throughout.
   - T64 GDN2: **6/5/6/5/5/6**, with P6 **482/469/480/474/478/487**.
   
   The approximate inverse-horizon decay is consistent with short-prefix dilution—not successful execution across the intervening targets. The first P6 threshold crossing is correctly distinguished from a monotone cliff, with independent fresh-pool sampling explicitly acknowledged.  
   Evidence: `results_stage21e_horizonladder/FINDINGS.md:18–45`; `REPORT.md:8–43`.

3. **Alignment and staging are reported honestly.** Audited native survivors at 2–32× are start-anchored with zero start offset; 1× mappings are ambiguous-coincident. There is no observed beyond-prefix transfer supporting suffix dominance, and structural alignment is unidentified. Registration also explicitly predicted these champions’ start anchoring—the conditional suffix hypothesis was not independently exercised by genuine new transfer. Curriculum beats cold four-to-two; wean ties two-to-two. No extending strong tail appears. Overlay/noncausal and T64 seed-confound disclosures are present.  
   Evidence: `REGISTRATION.md:16–33`; `FINDINGS.md:33–51`.

4. **Precision claims match implementation and receipts.**
   - Native state accumulation is fp32 despite bf16 autocast.
   - Quantization occurs after each token’s update, before readout and subsequent transition.
   - E97 native strong prefixes survive bf16/fp8/int8 through 32×.
   - GDN2 int8 losses and every champion’s already-1× int4 loss match the tables.
   - Wrapper/native discrepancies are disclosed, including per-column deltas and T64 GDN2 P6 **475 versus 482**. Maximum absolute deltas **1/2/7/1/1/4** were independently verified.
   - Integer budgets correctly include one fp32 scale per example/head/layer.
   
   These establish short-prefix precision thresholds, not a fitted bit-budget/horizon scaling law.  
   Evidence: `horizon_state_quantization.py:31–42,74–76,124–147,185–191`; `eval_horizon_ladder.py:333–339`; `FINDINGS.md:55–96`. Native accumulator evidence: `ndm/triton/e88_triton_forward.py:152`; FLA `ops/common/chunk_delta_h.py:81–95`.

### Required interpretation correction

**The requested “pair arriving at 2k−1/2k” premise is false for X21E.** Initial coordinates arrive **once**; subsequent inputs are drivers. Targets cumulatively follow
\[
(x,y)\mapsto(y,x+y^2+c)\pmod{61}.
\]
Intermediate targets are not fed back. X17 evaluation inserts neutral slots, not fresh state pairs.

Evidence:
- `eval_horizon_ladder.py:103–104`
- `run_invertible_stage2.py:92–93`
- `tasks/invertible_quadratic.py:25–46,61–70`
- `run_invertible_stage17_carrywean_8m.py:48–53,72–79`
- `INVERTIBLE_QUADRATIC.md:10–35,49–80`

**Exact acceptance condition:** retain the distinction that extending a *future suffix* cannot increase the computational depth of an already-tested causal prefix. Therefore prefix retention does not demonstrate long-range execution. **Do not conclude that this task family is position-local or inherently unable to measure long-range carry:** sufficiently accurate later-position predictions would test cumulative carry. The current FINDINGS appropriately limits its conclusion to the observed short prefixes; no measurement rerun is required.

### Read-only CPU validation

Parent ran:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/audit_horizon_ladder.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_horizon_ladder.py \
  tests/test_eval_invertible_stage18_freshdata.py \
  tests/test_invertible_quadratic.py
```

Results: artifact audit **PASS**; tests **66 passed, 10 CUDA tests skipped**. Supplemental independent checks verified **3,808 manifest hashes**, all survivor profiles, complete native alignment maps, precision deltas and metadata budgets. Supplemental assertions were corrected for disclosed GDN2 int8 survivors `[2,3,4]`, empty survivor sets, and nearest-candidate suffix-offset semantics—not data defects.

No issues found.

**Merge verdict: OK with notes.** No files modified; this is the complete read-only acceptance artifact.
---

## Parent attestation and interpretation correction (2026-10-22)

X21ER conditions satisfied. All three CPU audits PASS (artifact audit exit 0;
66 tests passed/10 CUDA skipped; supplemental: 3808 manifest hashes, 216 survivor
profiles, native zero-shift alignment maps, P6 [482,469,480,474,478,487], wrapper
deltas maxdelta [1,2,7,1,1,4], earliest eval 18:07:27Z after preregistration
18:01:09Z and freeze 18:07:11Z). The three supplemental-assertion corrections
(contiguity-from-P1, empty-survivor guard, suffix-offset semantics) are
documented audit-script assumptions, not data defects.

CONCESSION #17 (parent, interpretation): the parent's post-report claim that
"each driver position's answer is a function of the pair arriving immediately
before it - position-local" is FALSE for this task family. The task is
CUMULATIVE: initial coordinates arrive once; each driver demands the next state
of (x,y) -> (y, x+y^2+c) mod 61, so position k's answer requires carrying the
accumulated state across all preceding drivers (task sources: tasks/
invertible_quadratic.py:25-46,61-70; INVERTIBLE_QUADRATIC.md:10-35). The
correct statement, per the reviewer's exact condition: extending a future
suffix cannot increase the computational depth of an already-tested causal
prefix, so prefix retention does not demonstrate long-range execution - but
this task family CAN measure long-range carry (sufficiently accurate
later-position predictions would test cumulative state tracking); the models
simply never learned beyond a shallow iteration prefix (2-6 positions of the
3721-state quadratic-map automaton). FINDINGS.md's own limited conclusion
stands; no measurement rerun required.

Verdict: X21ER ACCEPT-WITH-CONDITIONS -> ACCEPTED with interpretation
correction receipted.
