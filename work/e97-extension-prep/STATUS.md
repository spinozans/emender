# E97 extension-prep work log (leg-2 worker)

Task: build extension-preparation-v1 (~1.8-2.2B tokens) for arc segments 9+ of
the v10 chain, while leg 1 awaits operator sign-off. CPU + bounded teacher-API
pipelines only; no GPUs, no CUDA, no training/serving/arc state. NEVER git add -A.

## Plan (execution order)
1. OH thinking enrichment: merge think-export.jsonl into the translated OH
   collection's Analysis channels (own recorded thinking; observe-then-quote
   adaptation where checkable; drop null/degenerate); audit; mask re-verify.
2. Hybrid-conversation cohort (~200-400 records) via teacher->real-Pi->verify
   (T2 curriculum-builder machinery): tool-flow records + pure-chat mirror,
   incl. the date-question specimen; deterministic outcome verification; zero
   protected-panel entity collisions.
3. Build extension-preparation-v1 at full pool strength with
   scripts/prepare_e97_pi_native_repair_training.py (whole-record 64K packing,
   interleaved emission; verify totals, cohort table, mask sums).
4. LR screen harness staged (not run): probe launcher + measurement plan for
   1e-5 / 3e-5 / 1e-4, 32-update probes on this prep.
5. Audit + scoped commits + final report. STAGED, NOT ADMITTED.

## Findings so far
- v2 prep consumed only slices: SmolTalk2 60M/650M targets, OH 2,413/5,087
  records. Full-strength extension = whole pools -> ~1.741B tokens /
  ~770.6M targets, within the ~1.8-2.2B / 800M-1B window once the OH-thinking
  delta and hybrid cohort land (pool ceilings cap it under 2B; no repetition
  will be invented to chase the top of the range).
- Scrub-report verdicts (the corrected scrub, T1): bridge + grounded-authored
  cohorts DROPPED (OH vocab 57%/80%, unreplayable); all other cohorts KEEP.
  v2 followed exactly this; the extension reuses v2's keep set at full strength.
- think-export alignment verified: 243,650 rows over 5,940 translated
  trajectories; 1,736 (tid,turn) pairs carry 2 rows each (reasoning_content +
  the think-pseudo-action thought of the same step); argument/action equality
  holds; the sealed 5,087 collection is a subset. 3,472 think-action rows map
  to steps whose on-wire turn is 'Think: null' with the thought only in
  Arguments - these merge candidates restore the model's own recorded thinking
  into the Analysis channel.
- Sealed OH collection already carries non-null Analysis on ~100% of
  assistant turns (verified on 300-record sample): T1's translator preserved
  reasoning_content. Step 1 is therefore: (a) re-verify exhaustively, not just
  sampled; (b) fill the think-action 'Think: null' turns from their own
  recorded thought rows; (c) cap to 2,048 analysis tokens; (d) drop degenerate
  fills; (e) re-verify mask alignment + frame validity.
- Teacher API: LunaRoute reachable (deepseek-4.1-flash available via
  gw.lunaroute.com/v1); auth via ~/.pi/agent/auth.json (session credential).

## Progress
- [x] Reading/verification pass complete (plans, T1 report, v2 manifest,
  preparer, curriculum builder, audits, transports).
- [x] Step 1 DONE (see /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-prep-v1/):
  - scripts/merge_e97_oh_recorded_thinking.py + tests (6 pass).
  - Merge audit: 203,240/203,240 assistant turns turn-aligned to
    think-export reasoning rows (0 mismatches); 1,464/1,464 think
    pseudo-action thoughts aligned; 0 fills needed (T1 translator already
    carried the model's own thinking on every turn); 0 degenerate; 0 over-cap.
  - Output payloads BYTE-IDENTICAL to sealed T1 collection (strongest mask
    re-verification); all frames parse as valid five-line Pi-native frames;
    spot checks render with non-null Analysis.
  - Observe-then-quote coverage measured: 97,592/198,113 post-observation
    turns (49.2%) quote the preceding observation. Thinking never rewritten.
  - Enriched authority at e97-extension-prep-v1/oh-thinking-enriched-authority
    (5,087 rec / 231,652,231 tok / 57,814,225 targets, training_eligible
    false).
- [ ] Step 2: hybrid-conversation cohort.
- [ ] Step 3: extension-preparation-v1 build.
- [ ] Step 4: LR screen harness staged.
- [ ] Step 5: audit + scoped commits + report.
