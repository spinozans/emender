# Quality Pass — E99 / typed Emender 1.3B LM-CMA launch batch

**Task:** `quality-pass-e99-1` (Evaluator role). Gate the E99 1.3B LM-CMA launch
batch *before* it runs: confirm the launch order, bound the "96" spend, lock the
candidate roster, and keep the final multi-day run behind an explicit human go.
This is a review-and-repair pass on downstream task descriptions, not new training.

## Batch shape (verified)

```
quality-pass-e99-1 (this task)
        │
   wire-e99-e98            sanity-wire all candidates into the real LM path
     ├──────────────┐
 run-e99-1-3b   run-matched-1-3b   96-eval LM-CMA top-up  +  matched controls
     └──────┬───────┘
   synthesize-e99-1-3b     decision + exact full-run SPEC (does NOT launch)
        │
   (human go required for any full multi-day 1.3B run — no such task exists yet)
```

Order is correct (wiring → top-up/controls → synthesis), dependency edges are
right, and there is **no rogue full-run task** in the graph. The chain terminates
at a written decision; launching the multi-day run requires an explicit human go.

## Grounding (real code, no invented infra)

- Real LM training entrypoint: `train.py` + `ndm/models/ladder_lm.py` (LadderLM),
  FLA-GDN via `ndm/models/fla_gated_delta.py` / `external_gdn2.py`. This is the
  same path that produced the E88 1.273B and FLA-GDN 1.352B baselines
  (`paper/review/THROUGHPUT.md`: E88 7,492 tok/s, FLA-GDN 8,248 tok/s, ctx 2048).
- Real held-out BPB pipeline: `scripts/measure_pile_bpb.py` →
  `paper/review/PILE_BPB_MEASURED.md`.
- E99 candidate origin: `typed-gdn-2-head` (native GDN-2/GatedDeltaNet recall heads
  + nonlinear specialist, 40:8 / 5:1 ratio; winner beat E98-CMA and an 8M DeltaNet
  ref on the synthetic suite). E98-CMA remains a serious Emender control and is
  **not** dismissed — synthesis must decide on LM evidence.

## Gaps found and repaired (edits applied to the four downstream tasks)

1. **Unbounded "96" / no hard stop.** `run-e99-1-3b` said "treat 96 as candidate
   evals, not full runs" but had no concrete compute cap. Added: per-candidate
   token/step + walltime cap (recorded up front, derived from wire-e99-e98
   projection), an aggregate GPU-hour ceiling, and a HARD STOP + honest log if the
   cost projection is wrong, instability/OOM occurs, or the ceiling is hit. The
   "promote top 2–3 to a longer pilot" step is now explicitly bounded (fixed
   multiple of short budget + walltime ceiling) so it cannot silently become the
   full run. `run-matched-1-3b` reuses the same caps for comparability.

2. **Checkpoint validation too weak.** All tasks asked only "checkpoint save/load
   works." `PILE_BPB_MEASURED.md` documents the exact failure that passes: an E88
   checkpoint that loaded **strict-clean (0 missing / 0 unexpected keys)** yet a
   standalone forward produced ~17.6 nats/token — worse than uniform-random
   (ln 50281 ≈ 10.83) — because the recurrence forward was structurally mismatched
   to how the weights were trained. Upgraded the criterion to **round-trip loss
   consistency**: reload in a fresh process and confirm the model reproduces the
   pre-save loss within tolerance; a load that does not is a hard blocker.

3. **No idle-GPU constraint.** Added "idle GPUs only; do not preempt other jobs"
   to the run tasks (matches the `typed-gdn-2-head` / throughput-measurement
   convention).

4. **Checkpoint publishing not explicitly forbidden.** "No HF publish" did not
   clearly cover staging/uploading generated checkpoints. Added explicit "do not
   publish/upload/stage any generated checkpoint" to all four tasks.

5. **Synthesis spec-vs-launch ambiguity.** Reinforced that `synthesize-e99-1-3b`
   produces a config SPEC for human review only and must not launch any
   long/full run; the human-go gate is restated in its validation.

## Validation checklist (this quality pass)

- [x] Downstream launch tasks have concrete bounded acceptance criteria and use
      real LM/Pile training (`train.py` + ladder_lm / FLA-GDN, `measure_pile_bpb`),
      not synthetic probes only — synthetic probes are guardrails, LM BPB decides.
- [x] "96" clarified as short-run candidate-eval top-up / budgeted LM-CMA, with
      per-candidate + aggregate caps and a hard stop/log if assumptions are wrong.
- [x] Candidate roster includes E99 GDN2+nonlinear, dense native GDN-2, current
      E98-CMA, and reused existing 1.3B controls (E88/GDN/M2RNN, provenance-labeled).
- [x] Final full 1.3B long run remains gated behind synthesis + explicit human go;
      no full-run task exists, synthesis emits a SPEC only.
- [x] No `paper/main.typ` edit, HF public release, push, or checkpoint publishing
      is authorized by any task in this batch.

## Residual risks (for downstream agents / human reviewer)

- Whether a *valid* 1.3B forward exists for E98-CMA / the typed mix in the current
  training path is unverified here — `wire-e99-e98` must prove it via round-trip
  loss consistency, and report an honest blocker (as in PILE_BPB_MEASURED) if not.
- "1.3B-class" sizing for the typed mix may not hit exactly 1.3B without breaking
  the 40:8 ratio; tasks permit the closest param-matched sizing table + a stop,
  rather than forcing the number.
