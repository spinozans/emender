# F9 staged deployment — operator restart required

Nothing here is installed into the live supply/collectors. Live teacher remains
lake-local; standing supply's source remains in the repo. The tracked teacher
patch reproduces the lake-local staged teacher without moving its ownership
into the repo. `prepare.py` builds a separate import overlay and versioned
source archive/registry; never deploys, admits, commits an allowlist or restarts.
The old `F9-staged/checkout` exploratory snapshot is superseded; use
`F9-staged/qualified-final` ONLY for historical Pi artifacts; the repaired
source closure for re-review is `F9-staged/qualified-F9G-final`.

## Reproduce without touching the daemon

On lambda01, using the project `.venv` (Frontier requires the canonical activation):

```bash
# CPU-only: the legacy wrapper below would run Pi and is NOT a CPU check.
# tests/test_f9_task_diversity.py requires separate operator Pi approval.
.venv/bin/python scripts/rl-loop-v1/F9-staged/prepare.py --root /new/private/F9-checkout
source /new/private/F9-checkout/activate.sh
cd "$F9_STAGE_ROOT"  # required: avoid importing live ndm from the working directory
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m pytest -q -p no:cacheprovider /home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py /home/erikg/emender/scripts/rl-loop-v1/F9-staged/test_f9r2_grammar.py -k 'not real_pi and not failed_reference_solution'
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python loop/rl_bank_unit_test.py --scratch /new/private/bank-fixtures
# Harvest is a separate owner-local operation, NOT needed for CPU checks.
```

Corpus is 8 admitted solved TRAIN bundles plus 4 invented fragments, owner-local
and immutable. Extract a single seed object from that local corpus into a new
file to use `teacher_author.py author --family terminal --seed <file>` or
`teacher_author.py morph --seed <solved-seed-file> --morph-trick entity_rename
--tranche <new-id>`. Morph choices are entity_rename, chain_depth, distractor,
failure_inversion. Author/morph commands make teacher API calls; qualification
used deterministic reference generators and made none. `build` produces only
quarantine. Full protected overlap/allowlist/admission/injection remains the
existing supply pipeline. The staged supply CLI refuses operational commands
without explicit `F9_DEPLOYMENT_APPROVED=1`; do not set it during validation.

## Coordinated operator deployment, not executed by F9

1. Review the staged diff and safety/test evidence. Stop supply/watchdog and
   affected collectors via the operator's normal attended procedure; never hot
   replace a module imported at tranche start. Preserve all old bank pins,
   source archives, validator programs, collection registries and pool tasks.
2. Materialize the additive scripts/provider and changed ndm modules from this
   directory into their corresponding repo paths. The archived checker and all
   era-2/3/4 validator bytes remain unchanged. Materialize staged loop changes
   into matching repo loop sources AND the operator's deployed loop copies.
   Deploy the staged `rl_bank_unit_test.py`: its three obsolete F8-only
   conversation-rejection assertions are replaced by real durable fresh-worker
   publication checks; the other 212 are unchanged. Integrated count remains
   **215**: explicitly check/update `run_bank.sh` preflight literal to 215 at
   deployment (repo and deployed wrapper), never leave it at 185 or infer it.
   Put patched `teacher_author.py` ONLY into lake `scripts/teacher_author.py`,
   and `inject_pool.py` into lake `scripts/inject_pool.py`. Standing supply stays
   repo-owned (its lake launch copy, if used, is an operator deployment copy).
3. Rebuild/verify the complete versioned generator manifest/archive and source
   registry using the committed deployed bytes. Retain the old authority under
   its old hashes. Both first-party registry source_archive pins must match the
   new archive. Do not copy an exploratory pre-commit generation receipt.
   The operator allowlist remains the canonical checked-in trust root, not a
   staged snapshot. No new collection may bypass an authorized overlap tuple.
4. Configure durable environment for supply, watchdog and collectors:
   `F9_STAGE_ROOT=/home/erikg/emender`,
   `F9_AUTHOR_WORK=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1`,
   and supply-only `F9_DEPLOYMENT_APPROVED=1`. The stage-root name also selects
   the versioned validator/provider. Ensure deployed lake and loop imports use
   the matching versions; no collector may receive new-family tasks before its
   safe bridge and full-exchange/end-state grader are installed. Carry these
   settings in the operator's persistent launch/watchdog environment, not just
   an interactive shell. Do not change the standing lane/quota knobs in F9.
5. Run isolated fresh probes against the committed deployment; confirm old
   pins/legacy finish remain unchanged. Restart only on operator approval.
   Watch per-family pass/fail counts, sealed script rejections, invalid-action
   failures, fresh-correction volume and PG contrast before increasing shares.

The shell gate is REQUIRED on both proof and policy/teacher attempts. It is a
finite offline grammar, not an OS sandbox; bwrap is absent on lambda01. It
rejects arbitrary interpreters, substitutions, absolute/traversal paths,
networking, background commands and private git metadata before dispatch.
Only the approved file-processing/local-git forms ship. No unconstrained
terminal execution, arbitrary package/environment installation or executable
fixture runner is authorized by this deployment. Extending that grammar or
qualifying bwrap is a separately reviewed safety/coverage change.

## Attended release/rollback runbook (NOT executed by F9C)

Do not run this section during ongoing collection. Deployment is **NO-GO**
until independent review of the F9E exact-byte closure and separate operator
approval. Run on lambda01 with `PY=/home/erikg/emender/.venv/bin/python`;
Frontier instead requires canonical activation and `PY="$EMENDER_PYTHON"`.
Set `REPO=/home/erikg/emender`, `LOOP=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1`,
`LAKE=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1`.
Set `F9_TOOLS="$REPO/scripts/rl-loop-v1/F9-staged"` before either boundary
sequence; this retained release-tool directory contains `rollback_gate.py`.
Record its absolute path and tool hashes in `release.manifest.json` and retain
it through rollback (do not substitute restored old-only runtime tools).
`BANK` MUST be the operator's current explicit bank path (not a guessed default).
`RELEASE` is a new owner-private directory outside all three roots. Maintain:

* `backup.manifest.json`: absolute destination, relative saved payload, SHA256
  for **every replaced file**, and `absent_before` for F9-added source files.
* `release.manifest.json`: same exact destination roster and SHA256 of approved
  committed release payload, commit ID, source-archive/manifest/registry digests,
  seed pins, integrated evidence hash and interpreter identity.
* `launch.manifest.json` plus `old-launch.env` / `new-launch.env`: exact launch
  commands/argv, environment and cwd for supply, watchdog, bank owner,
  collector and correction workers; absolute module import paths and hashes;
  persistent launcher/cron paths; each owned PID, `/proc/PID/stat` birth and PGID.
  Identify the ONE stream publisher for the attended bank/lane. These are
  deployment inputs, not historical PID assertions from F9-evidence.json.
* `cron.before`, `cron.disabled`, `imports.before.json`, `imports.release.json`,
  `boundary.before.json`, `boundary.after.json`, `first-injection.json` (write
  immutable task IDs/time/authority hashes BEFORE first new-family injection).

Roster (same-name files map to destination roots): staged `scripts/*` -> repo
`scripts/*`; `ndm/*` -> repo `ndm/*`; scripted provider -> repo `configs/pi/`;
`rl_bank_lane.py`, `rl_correction_{service,pool,config}.py`, suite -> BOTH repo
`scripts/rl-loop-v1/` and `$LOOP/scripts/`; driver ->
`$LOOP/scripts/rl_loop_driver.py`; teacher -> `$LAKE/scripts/teacher_author.py`;
injector -> `$LAKE/scripts/inject_pool.py`; supply -> repo loop path and any
lake launcher copy recorded in launch.manifest.json. Preserve CURRENT bank
helpers/coordinator/receipts/deadlines and all supply knobs. Also roster all
mutable generator manifests/registry/current authority pointers and persistent
launch environment/wrappers, including `run_bank.sh` and its **215** preflight.
Do NOT roster immutable receipts, archived authority, pinned validators, spool
jobs, pool tasks, stream rows or checkpoints as restorable/deletable code.
Archive prior authority by digest before changing mutable registry pointers.

### Stop, drain and establish exclusive ownership (both directions)

1. **NO-GO if any launcher/owner is unknown.** Save `crontab -l > "$RELEASE/cron.before"`.
   Remove ONLY the recorded standing-supply watchdog line into cron.disabled;
   install with `crontab "$RELEASE/cron.disabled"`. Disable any other recorded
   auto-relauncher in the launch manifest. Acquire and HOLD the watchdog lock
   (`$LAKE/standing-supply/watchdog.lock`) with `flock -n` for the entire stop,
   restore, verification and initial restart interval. A lock failure is ABORT:
   an existing watchdog may be running. Never merely kill supply with cron armed.
2. Stop new supply authoring/admission/injection via the owned process group.
   After checking each saved PID's birth/cmdline/PGID matches the manifest,
   `kill -TERM -- -"$SUPPLY_PGID"`; wait to an explicit operator deadline;
   escalate to KILL only for that verified group, record interrupted tranche as
   quarantined, never admit it. Unknown or reused PID is ABORT, not kill-by-name.
3. `touch "$BANK/STOP"` and wait for collector/owner to finish cycle/publication
   to the existing bounded shutdown deadline (1800s in run_bank.sh). STOP stops
   admission/generation, NOT the durable correction service. Freeze pool
   refresh, claim sweeping, learner stream writes and any extra launcher.
   If the owner exited with correction backlog, authorize exactly ONE attended
   compatible CPU publisher using `publish_corrections(Path(BANK), lane)`;
   never run it concurrently with collector/daemon publication. Record ownership.
4. **Prefer drain:** retain compatible code/import environment while the service
   completes queued jobs; publish ready jobs with the sole lane owner. Inventory
   each `lanes/lane-NN/corrections/job-*/{spec,execution,result,publication,published}.json`
   and all pending/claimed bank task IDs. Claims delegated to a durable job are
   never manually swept/requeued/TTL-recycled. Correct claims leave only via
   compatible publication/explicit failure retirement. Async OFF only prevents
   new admissions; it neither stops nor deletes existing jobs.
5. Stop the durable manager SEPARATELY after drain: verify manager PID birth,
   command bank/lane and its singleton `service.lock` owner; TERM that owned
   manager, wait a finite 10s, KILL only its verified group if needed. Account
   for pool worker/Pi groups (workers have separate PGIDs; do not assume killing
   collector killed service). On bounded interruption, retained execution claims
   are failed explicitly by compatible service restart; queued jobs resume,
   ready results/intents are replayed once. Never rerun a mutated interrupted
   splice. Reconcile result/publication/retirement before handing off ownership.
6. Prove all owned groups exited and all lane service locks are freely acquirable
   with nonblocking flock, then release probe locks (keep watchdog lock held).
   No source replacement while any service worker/publisher imports it.
   ABORT on live unknown owner, partial stream tail, digest/routing mismatch,
   unavailable spool or incomplete shutdown. Stay stopped; retain evidence.

### Back up and verify exact bytes (before any replacement)

With owners stopped, create backup payloads using the exact release destination
roster, not a directory-wide copy of bank state. The manifest schema is
`{"files":[{"destination":"/absolute/path","payload":"payload/relative/path",
"sha256":"hex","absent_before":false}, ...]}`; for absent source paths use
`absent_before:true` and no payload/digest. Payload paths must remain within
RELEASE. Save old authority archives and pins independently under their hashes.
Record/import-check in a NEW CPU interpreter from each launcher cwd/environment:
`rl_bank_lane`, `rl_loop_driver`, `rl_correction_service`, `rl_correction_pool`,
`rl_correction_config`, `standing_supply`, `teacher_author`, `inject_pool`,
`ndm.e97_first_party_source_archive`, `scripts.e97_offline_shell` and bridges.
For old imports, F9-only modules are recorded absent, never spuriously required.
Example exact path/hash probe (no episode or service start):

```bash
cd "$RECORDED_CWD"
# Source the recorded old/new launch env; do not guess PYTHONPATH.
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import hashlib, importlib, json, os
out = {}
for name in os.environ['PROBE_MODULES'].split(','):
    p = importlib.import_module(name).__file__
    out[name] = {'path': p, 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()}
print(json.dumps(out, sort_keys=True, indent=2))
PY
```

Compare EXACT resolved paths and digests with the manifest, including every
runtime helper from F9-evidence's integrated roster, not just top-level imports.
**NO-GO** if import paths fall through to stale deployed modules. Rebuild the
archive from committed bytes and verify 17 legacy or 18 additive components,
old-validator member equality, both first-party registry archive pins, and seed
intake pins. Run isolated old/new probes and CPU suites before any injection.
The repaired staged suite is **215**, including the three real fresh-worker
checks replacing live F8 rejection tests. Deploy that same flip to the LIVE
suite and verify BOTH wrapper preflight literals explicitly equal 215. Run the
F9 CPU selection separately; Pi proof qualification requires operator approval.

### Rollback BEFORE first new-family injection

1. Require no `first-injection.json`, confirm supply disabled and STOP held.
   From the repaired staged tools, run the READ-ONLY gate:
   `"$PY" "$F9_TOOLS/rollback_gate.py" --bank "$BANK" > "$RELEASE/boundary.before.json"`.
   GO only on exit 0: zero new-family pending/claims AND zero unpublished durable
   jobs. Hash drift, missing layout, unknown records or nonzero exit = ABORT;
   use post-injection compatibility path instead. A missing marker alone is
   never evidence that no work was injected. Recheck immediately before restore.
2. Restore only backup.manifest destinations, with this stopped-source command:

```bash
RELEASE="$RELEASE" PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
r = Path(os.environ['RELEASE']).resolve()
m = json.loads((r / 'backup.manifest.json').read_bytes())
# Verify ALL backups before changing ANY destination.
for f in m['files']:
    if not f['absent_before']:
        p = (r / f['payload']).resolve()
        assert p.is_relative_to(r) and hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256']
for f in m['files']:
    dst = Path(f['destination'])
    if f['absent_before']:
        # Roster must authorize only F9-added source/provider files, never data.
        assert dst.suffix in {'.py', '.ts'}
        dst.unlink(missing_ok=True)
    else:
        tmp = dst.with_name(dst.name + '.rollback-tmp')
        shutil.copy2(r / f['payload'], tmp)
        os.replace(tmp, dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == f['sha256']
PY
```

3. Restore `old-launch.env` and saved mutable authority/current registry from
   manifest; do NOT delete new archived authority or immutable pins. Verify all
   restored hashes and exact imports match imports.before.json in fresh
   interpreters, verify old pins recover byte-identical validators, and run old
   suite with its own matching preflight literal. Failure = remain STOPPED.
4. Restart in order: compatible correction service via the sole lane owner
   (locks prove single manager), publisher and collector using recorded old
   argv/env, then learner/coordinator as authorized, then old supply. Keep
   injection fenced until old-task isolated probe succeeds. `run_bank.sh` removes
   STOP itself: do NOT invoke it prematurely. Re-arm ONLY the saved watchdog
   cron/launch environment after process/import/heartbeat/source-hash checks;
   release watchdog lock last and verify relaunch uses old paths/environment.
   ABORT on unexpected restart/claim/receipt, wrong hash, or unsafe bridge call.

### Rollback AFTER injection (old-only restore is UNSAFE)

1. Execute the same stop/drain procedure; retain the repaired compatible release
   bytes and all new authority/pins. The boundary gate enumerates new pending/
   claimed work and unpublished correction jobs. Save that immutable inventory.
   No old-only grading, sweeper, publisher or collector may touch that bank yet.
2. Preferred completion path: with supply/refresh/injection/watchdog disabled,
   authorize the ONE compatible lane owner to complete already pending/claimed
   new tasks and publish/retire ALL durable jobs under their original pins. Run
   only with repaired validator/bridge/driver and current correction service.
   No automatic retry of interrupted execution claims. Repeat inventories until
   gate exits 0, locks are free, task claims reconciled and streams validate.
   Then follow pre-injection restore steps 2–4. A failed grade is explicit failure
   retirement, not a receipt or deletion. Preserve publication and stream chain.
3. If completion cannot be authorized, **isolate the ENTIRE affected bank**, not
   selected job directories/claims (absolute owned workspace paths are sealed).
   Leave its path and immutable contents in place, STOP held, all launchers/
   refresh/sweep/publication disabled; retain a complete compatible code/env
   release for future attended completion. Do not repoint archived spec paths.
   Run old-only code, if separately approved, solely against a NEW empty bank
   with verified old-only task selection and no learner reading isolated streams.
   No restoration against the affected bank until its boundary gate passes.
   Missing isolation proof or unresolved ownership = ABORT, remain offline.
4. Immutable receipts, job specs/results/publication intents, pinned validators,
   source archives and admitted task evidence are NEVER deleted, edited or
   renamed out of their bound paths to make the gate pass. Monitor invalid-action
   failures, fresh-correction routing, per-family grades and spool growth after
   restart. Any wrong import/hash, raw expansion admitted, unknown owner, mixed
   authority or duplicate claim/publication is immediate stop/rollback assessment.

CPU boundary test: `unit_tests.py::test_rollback_boundary_is_read_only_and_refuses_inflight_work`
checks before-injection GO, pending and claimed conversation NO-GO, unpublished
jobs NO-GO, retained published evidence GO, byte preservation and hash-drift
failure. The integrated 215 suite exercises real durable manager/worker loss,
queued resume and idempotent publication. This is not a live rollback rehearsal;
actual launcher ownership, restart and watchdog race proof are deploy-time gates.

### Seed corpus deployment ownership (C1)

The operator owns the corpus under `$F9_AUTHOR_WORK/seed-corpus.json` with the
sibling `seed-manifest.json`, not a relative cwd file or a teacher download.
Copy the retained owner-local corpus from historical
`$LAKE/F9-staged/qualified-final/author-work/` only AFTER verifying against the
reviewed release pins. Install both read-only, e.g. `install -m 0400`, without
modifying the historical originals; refuse an existing different corpus.
For this release the corpus SHA256 is
`1855368bf374849e94c2bfd6fb46b4ef36f025a5b509e04ce7819ef30568d101`, and manifest
SHA256 is `bfa5e4418afd8f20aeaaccbe44cafcec413baa568cc904c988e5878b5d950a13`.
Record both in release.manifest.json; carry
`F9_SEED_MANIFEST_SHA256=bfa5e4418afd8f20aeaaccbe44cafcec413baa568cc904c988e5878b5d950a13`
in supply AND watchdog's persistent launch environment. Supply verifies the
manifest pin, corpus hash, item hashes/count/provenance and ALL solved entries
via verify_seed (TRAIN admission/proof/bundle/prompt/fixtures) before constructing
any seeded instruction. Entries remain verified in memory for that tranche.
Absent/unpinned/corrupt/unauthorized corpus explicitly disables seeded authoring,
prints `SEED_INTAKE disabled: <reason>` and writes tranche `seed-intake.log`;
there is no silent fallback. Verified intake logs count and manifest pin.
Unseeded authoring retains all candidate gates. If seeded deployment was
intended, a disable log is NO-GO for ramp-up: stop supply, repair ownership/pins
and requalify. Never recompute a pin to bless unreviewed bytes.

Final CPU closure: **215 bank checks** + **74 F9 CPU tests** (the original 42
plus 32 repair checks); 3 Pi-dependent tests remain excluded. This count does
NOT change bank preflight's literal 215. No new live Pi/API/GPU proof or rollback
rehearsal is asserted.
