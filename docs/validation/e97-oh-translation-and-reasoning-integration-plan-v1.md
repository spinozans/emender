# E97 OH-Translation + Reasoning-Rehearsal Integration Plan (v9-full arc restart)

Status: ACTIVE — authorized by operator 2026-09-18 ("let's go into this").
This document is the durable authority for the three-task dependency graph below.
All three tasks read this doc; the integration task (T3) requires T1 and T2.

## Background

The v9-full wide arc stopped fail-closed at segment 1 (u128 Stage-B 3/14 valid)
because the full-prep OpenHands cohort carried raw OH-protocol text
(`Analysis:/Commentary:/Think:` scaffolds, `str_replace_editor`,
`execute_bash`) in ~90% of sampled records; at 17.1% of targets it captured
the model's workspace-task behavior. Full forensics in
`docs/validation/e97-pi-native-tool-copy-progress.md` (search "OpenHands
protocol capture"). The arc does not restart until the data authority is fixed.

## Dependency graph

```
T1: OH -> Pi-native translation + replay verification  ─┐
   (parallel, long: up to ~8h, CPU + containers)        ├─> T3: Integration
T2: Reasoning-rehearsal cohort spec + bounded pilot     ─┘   (re-prep, key screen,
   (parallel, ~1-2h, CPU + bounded teacher pilot)            freeze proposals, stage
                                                              admissions; NO launch
                                                              without operator sign-off)
```

## T1 contract — OH→Pi-native translation, fail-closed by replay

1. Locate the raw OH trajectory sources that fed the preparer's
   `openhands-execution-rehearsal` cohort (trace `source_row`/`source_record_id`
   provenance in the prep records; check the preparer's OH ingestion path in
   `scripts/prepare_e97_pi_native_repair_training.py` and the OH source
   manifests under `/mnt/nvme1n1/erikg/sft/`).
2. Build `scripts/build_e97_oh_pi_native_translation.py`: re-render every OH
   trajectory as Pi-native records under the canonical Pi-core system prompt,
   with the mechanical action mapping:
   - `str_replace_editor` view -> `read` (view_range -> offset/limit)
   - `str_replace_editor` create -> `write`
   - `str_replace_editor` edit -> `edit` (old_text/new_text -> oldText/newText;
     identical str_replace contract)
   - `execute_bash` -> `bash`
   - OH observations -> Pi observation rendering (verbatim file/command
     content, Pi wrapper format)
   - OH `think` actions: EXPORT SEPARATELY as grounded pre-action commentary
     records for T2 (do not discard); null scaffolds are dropped, never
     translated to literal "null" text.
3. **Replay verification is mandatory for every record that enters the
   collection**: execute the translated Pi actions against the recorded initial
   repo state in a container (existing sandbox images / OpenHands runtime
   images; persistent parallel worker pool, CPU-only — GPUs are off-limits),
   require the final file state to match the recorded final state exactly, else
   DROP the record. No record ships without replay PASS. If the pace projects
   >10h at the pool size, report back with measurements — do not silently
   downgrade to sampling.
4. Scrub pass over ALL other prep cohorts: scan ASSISTANT TARGETS for
   OH-protocol markers; fix (if mechanically translatable) or drop those
   records; report counts before/after.
5. Deliverables: translator + verifier scripts, translated+replay-verified
   candidate-authority collection (manifest + records.jsonl, Pi-native),
   think-export for T2, scrub report, tests, scoped commits, and a report with
   counts: pool size, translated, replay-passed, dropped, marker deltas.

## T2 contract — reasoning-rehearsal cohort (thinking traces)

1. Design doc `docs/validation/e97-reasoning-rehearsal-cohort-spec-v1.md`:
   - Thinking channel = bounded plain-assistant commentary BEFORE the action
     line (no new tokens; grounded style: quote observations, state short
     plans; no long metaphysics — this is a 4B model).
   - Codec/contract analysis: determine whether pre-action commentary breaks
     the Stage-B "valid first frame" contract (read
     `ndm/e97_agent_protocol.py` + the stage-b frame checker); if it does,
     specify the bounded-commentary amendment as an explicit, evaluated
     contract change.
   - Supervision rule: teacher thinking is supervised ONLY on
     deterministically-verified successful trajectories (existing
     masked-failure discipline extended one channel); thinking text is never
     trusted on its own.
   - Target family: multi-step composition (the 4/16 axis) — explicit
     step-plans across chained actions.
   - Sources: (a) teacher proposals with pre-action reasoning via the existing
     DeepSeek→real-Pi→verify pipeline; (b) T1's think-export; (c) STaR-style
     self-sampling noted as a follow-up (needs the served model).
2. Bounded pilot: 50 records through the real teacher→real-Pi→verifier
   pipeline with pre-action commentary rendered Pi-natively; deterministic
   outcome verification for every record; audit like prior collections
   (zero exact-entity collisions with protected panels).
3. Deliverables: spec doc, prompt templates, pilot collection + audit,
   tests, scoped commits.

## T3 contract — integration (REQUIRES T1 + T2 outputs)

1. Build the revised cohort-spec set: translated+replay-verified OH collection
   REPLACES the raw OH cohort; T2's pilot reasoning cohort enters as a minor
   cohort; all other cohorts as scrubbed by T1.
2. Re-run the preparer for the revised full prep; verify pack/record/token
   totals; screen keys under the window-coverage rule (majors >=2% every
   update, minors every 32 updates) until 8 keys are admitted.
3. Freeze 8 segment proposals (128 updates each, keys in screened order,
   bridge parent for seg1, window-coverage binding), audit each with the
   generic auditor, commit+push.
4. STAGE admissions and launchers (do not execute admissions without operator
   sign-off — new data authority requires explicit authorization).
5. Deliverable: integration summary doc listing every gate the operator signs
   off: prep identity, cohort table (with scrub + replay counts), keys,
   proposals+audits, admission commands, arc launch plan, and the unchanged
   evaluation protocol (juncture detectors at 128-update boundaries; full
   frozen dual gates at u256/u512/u896/u1024; fail-closed stopping armed).

## Standing constraints for all tasks

- GPUs are for the interactive server (1 leased GPU) and nothing else unless
  the operator says otherwise; T1/T2/T3 are CPU-only (containers OK).
- Scoped commits only; the tree has extensive unrelated dirty files; never
  `git add -A`.
- No threshold changes; no automatic retries; evaluation protocol unchanged.
- All LunarRoute-backed subagents (current session model inherited).
