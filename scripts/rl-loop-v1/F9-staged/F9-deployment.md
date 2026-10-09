# F9 staged deployment — operator restart required

Nothing here is installed into the live supply/collectors. Live teacher remains
lake-local; standing supply's source remains in the repo. The tracked teacher
patch reproduces the lake-local staged teacher without moving its ownership
into the repo. `prepare.py` builds a separate import overlay and versioned
source archive/registry; never deploys, admits, commits an allowlist or restarts.
The old `F9-staged/checkout` exploratory snapshot is superseded; use
`F9-staged/qualified-final` for the retained qualification evidence.

## Reproduce without touching the daemon

On lambda01, using the project `.venv` (Frontier requires the canonical activation):

```bash
.venv/bin/python -m pytest -q tests/test_f9_task_diversity.py
.venv/bin/python scripts/rl-loop-v1/F9-staged/prepare.py --root /new/private/F9-checkout
source /new/private/F9-checkout/activate.sh
cd "$F9_STAGE_ROOT"  # required: avoid importing live ndm from the working directory
/home/erikg/emender/.venv/bin/python -m pytest -q /home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py
/home/erikg/emender/.venv/bin/python /home/erikg/emender/scripts/rl-loop-v1/F9-staged/harvest_seeds.py
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
