# T1 report — OH → Pi-native translation with fail-closed replay verification

Task: section "T1 contract" of
`docs/validation/e97-oh-translation-and-reasoning-integration-plan-v1.md`.
Work dir: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-oh-pi-native-translation-v1/`.

## Headline counts

| Stage | Count |
|---|---|
| Raw OH trajectory pool (`e97-4b-open-swe-source-native-fulltraj-v1`) | 5,969 |
| Translated to canonical `e97-pi-native-v1` records | 5,940 (29 dropped: 26 invalid view_range errors, 3 malformed create args) |
| Replay candidates | 5,940 |
| Replay PASSED (full fail-closed gates) | 5,191 |
| — of which re-admitted after dir-listing adjudication | 10 |
| Replay DROPPED (final) | 749 |
| Sealed collection records (≤64K context) | 5,087 (104 excluded over context) |
| Sealed tokens / supervised targets | 231,652,231 / 57,814,225 |
| Think-export rows for T2 | 243,650 |

Drop taxonomy (final 749): read_missing_file 332, read_content_mismatch 141,
final_state_mismatch 125, patch_apply_failed 31, edit_missing_file 25,
edit_not_unique 20, read_is_directory 20, checkout_failed 8, clone_failed 6,
exception 9, read_binary 3, read_offset_error 4, write_existing_file 19,
dir_listing_mismatch 0 (all adjudicated). These are genuine divergences
between the recorded OH trajectory and the recorded repository state — exactly
the fail-closed contract: not reproducible ⇒ not shipped.

## The replay gate (what every PASSED record proves)

1. The repo state at `base_commit` (from the SWE-rebench metadata join; 5,969/5,969
   coverage) was materialized with full git history.
2. Every mapped action executed in order: file ops on the worktree, bash inside
   a docker container (`--user`, no bytecode) mounted at the OH path convention.
3. `read` observations verified line-by-line against the actual file (OH `cat -n`
   format parsed); dataset pre-clip markers verified as verified-prefix only.
4. `edit` oldText uniqueness + exact application; `write` no-clobber; failed
   OH actions verified to fail identically in replay and shipped as authentic
   error observations (isError=true) — recovery data, not corruption.
5. `bash` outputs verified for the deterministic class (order-insensitive for
   enumerations); environment-dependent commands executed for state fidelity and
   shipped as recorded-verbatim authentic observations (counted in stats).
6. **Final-state gate**: every file touched by the recorded model patch (applied
   to a pristine base) must equal the replayed tree's file, exactly.

## Documented relaxations of the dir-listing observation check (auditable)

The supervisor's rule: path-set fidelity, timing/normalization checks before
calling corruption, document every relaxation. Three normalizations were
introduced, each with captured evidence:

1. **Symlink traversal** (4 records re-admitted: 2× osbuild `samples`,
   2× electronim `assets`; plus 3× dtolnay/cxx from the pilot set).
   Evidence: `git ls-tree -l <base> samples` → mode `120000` (symlink) for
   osbuild; dtolnay/cxx `gen/src/include` is a symlink to `../../include` —
   the recorded OH listing contains `gen/src/include/cxx.h` (OH's view follows
   symlinks) while GNU find without `-L` lists the link but does not descend.
   The check now classifies recorded paths whose ancestor is a symlink as
   rendering artifacts (counted in `dir_listing_rendering_artifacts`), and the
   translator's mapping for future runs emits `find -L`. All 10 re-admitted
   records then passed the FULL gate set including the exact final-state gate.

2. **Hidden-dir find roots** (2 records re-admitted: ueli `.vscode`,
   plexis `.package-template`). The OH view itself targeted a hidden directory;
   the mapped find's `-not -path '*/.*'` clause excludes everything under it by
   construction, so the executed listing is structurally empty. Hidden is now
   judged against the `/workspace` root, not the find root.

3. **Trailing-slash filter bug in my own check** (1 record re-admitted:
   `996562a8`, pandas). `_listing_paths` filtered on the un-normalized form, so
   the executed find's bare `/workspace` start-point line was dropped while the
   recorded trailing-slash form survived — an artifact of MY check, not the
   data. Fixed the filter; the record then passed everything.

No re-admission occurred without the record passing every gate, including the
exact final-state oracle. Genuine path-set discrepancies at the same trajectory
point were kept dropped (none of the final 749 contain dir-listing failures:
the 2 osbuild/electronim assets-and-samples cases were explained by symlinks;
nothing remained).

## Scrub pass (assistant-target OH-vocabulary markers)

`scrub-report.json`: fulltraj cohort 100% OH-vocab (replaced by this
collection); grounded-authored 80% and representation-bridge 57% (DROP
recommended — no retained workspaces/base commits, replay verification
impossible under the fail-closed rule); all other cohorts clean (0 hits in
sampled supervised targets): instruction 250K, conversation 620K, compositional
41K, cumulative 61K, live-aligned 20K, curriculum 2,012, plus
pointerchase/longcopy/extracterror.

## Think-export for T2

`think-export.jsonl`: 243,650 grounded pre-action commentary rows
(trajectory_id, turn, reasoning_content, action, arguments) — aligned to
replay-verified trajectories.

## Scripts, tests, commits

- `scripts/build_e97_oh_pi_native_translation.py` (translator; `-L` mapping fix)
- `scripts/verify_e97_oh_translation_replay.py` (fail-closed verifier + instrumented
  entry points; `_listing_paths` fix; symlink/hidden normalization; thread-safe
  debug capture)
- `scripts/adjudicate_e97_dir_listing_drops.py` (drop adjudication tool)
- `scripts/build_e97_oh_verified_authority.py` (piecewise-linear collection assembler)
- `tests/test_build_e97_oh_pi_native_translation.py` (8 tests)

## Disk usage

- Work dir total ≈ 43 GB (repo-cache 39G — 1,122 bare clones, reusable for T2/T3
  and any future re-verification; candidates 1.6G; sealed collection 1.1G;
  replay verified 805M; swe-rebench metadata 415M; think-export 217M;
  model-patches oracle cache 24M).
- No GPU used at any point.

## Collection location

`candidate-authority/`: manifest.json (schema emender-e97-tulu3-masked-sft-v1,
`training_eligible: false` — admission flips it), tokens.uint32.bin,
assistant_mask.uint8.bin (verified mask alignment), records.idx, records.jsonl.
Oracle patches were used strictly as verification oracles and never entered the
payload (`oracle_metadata_copied: false`).
