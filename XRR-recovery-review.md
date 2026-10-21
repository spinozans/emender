## Review

**Verdict: BLOCKED pending C1–C2 below.** The split-provenance admission is coherent, but one unreceipted evidence-retention deviation prevents unconditional acceptance.

Paths below are relative to `experiments/expressivity_tasks/`.

### Correct

- **Originals and restart boundaries are explicitly preserved.** Per-run amendments and supersessions identify checkpoint paths/SHA256, full versus partial interruption rows, new output namespaces, and unchanged scientific schedules.
- **SPLIT provenance is consistent:**
  - `e97_w64`, `gdn2_w512`, `gdn2_w2048` retain `infra_resume/*/replay_verified.json`. Their confirmation amendments explicitly limit verification to diagnostics; empty partial-row diagnostics are not represented as full comparisons.
  - `e97_w32`, curriculum, stage17 use separate `infra_restart` admissions. Their amendments, supersessions, and per-update provenance explicitly deny at-death restoration.
  - Stage15 amendment-2 likewise denies at-death restoration and preserves the original **1M** cosine/ceiling; the erroneous task-memory 500K endpoint is explicitly corrected.
- **Exact gates failed closed, without tolerance:**

  | Arm | Rejected diagnostic |
  |---|---|
  | Stage15 Muon | `muon_update_rms`, 433044, difference ≈5.23e-11 |
  | e97_w32 | `grad_norm`, 245104 |
  | Curriculum | `grad_norm`, 252204 |
  | Stage17 | `loss`, 160022 |

  I read each `interrupted.json`, terminal receipt and `.FAIL` marker; statuses agree. The failed admissions remain failed evidence, not retrospectively relabeled successes.
- **200GiB guards and tests exist.** See `scripts/invertible_stage16_17_resume.py:25–44`, shared by its restart runner; and the stage15 guard/restart tests. Committed prelaunch receipts report **54 passed** for stages16/17 and **28 passed, 1 skipped** for stage15. These are receipts reviewed, not tests independently executed here.

### Finding — P1: Stage15 restart silently omits pruning receipts

`scripts/invertible_stage15_muon_restart.py:159–163` directly deletes superseded checkpoints with `unlink()`, without deletion/hash receipts.

This contradicts `results_invertible_stage15_probes_8m/REGISTRATION.md:58–59`: **“pruning receipts retain hashes.”** Amendment-2 permits pruning new restart checkpoints but does not receipt removal of this evidence requirement.

This is already observable: `muon_restart/muon3x_seed0/checkpoints.json:2202–2205` records `step_435000.pt` and its SHA256, but the checkpoint is absent and the directory contains no `pruning.jsonl`. Index-recorded hashes are not deletion-time verification receipts.

**Smallest corrective action:** an additive, operator-approved concession/remediation receipt—not rewriting frozen registrations or fabricating historical deletion verification.

### Provenance correction

The supervisor supplied historical verification correcting the task’s commit attribution: **3562166a is a whitespace-only growth-snapshot change**, not the recovery implementation. The relevant recovery chain is `cce1d03e → 0e44cdaf / 2821b2ed → 2d135738`. This review relies on that supplied historical evidence; my tools cannot execute `git show/log`.

### Exact acceptance conditions

**C1:**  
> Commit an operator-approved append-only X15 amendment explicitly acknowledging omitted checkpoint-deletion receipts, inventorying missing restart checkpoints with their index-recorded hashes, distinguishing those hashes from deletion-time verification, and authorizing a prospective remedy or explicit evidence concession. Preserve all original registrations, failed replay receipts, and already-bound runner bytes. Do not automatically stop, restart, or relaunch training.

**C2:**  
> Run and retain the read-only CPU audit results below. All registration/source/artifact/checkpoint bindings, replay comparisons, and preserved FAIL receipts must verify. Resolve every unexpected failure explicitly before acceptance.

**Merge verdict: BLOCK.** This does not itself authorize intervention in running jobs. Training outcomes remain censored until their registered terminal gates.

## Requested CPU audits

On **lambda01**, from `/home/erikg/emender`:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_invertible_stage15_muon_resume.py \
  tests/test_invertible_stage15_muon_restart.py \
  tests/test_invertible_stage15_probes_8m.py \
  tests/test_invertible_stage15_amendment.py \
  tests/test_invertible_stage16_17_resume.py \
  tests/test_invertible_stage16_17_restart.py \
  tests/test_invertible_stage16_ignition_8m.py \
  tests/test_invertible_stage17_carrywean_8m.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts import invertible_stage16_17_resume as r
from scripts import invertible_stage16_17_restart as s
from scripts import invertible_stage15_muon_restart as m

read = lambda p: json.loads(Path(p).read_text())

for name in r.BINDINGS:
    reg = s.verify(name) if name in s.BINDINGS else r.verify(name)
    cp, _ = r.checkpoint_preflight(r.module_for(name).ROOT / name)
    assert cp == reg["resume"][name]["checkpoint"]
    old = r.root_for(name)
    if name in s.BINDINGS:
        failure = read(old / name / "interrupted.json")
        terminal = read(old / "receipts" / name / "terminal.json")
        assert terminal == read(old / "markers" / (name + ".FAIL"))
        assert terminal["exit_code"] != 0
        assert terminal["status"] == "INFRA-INTERRUPTED"
        assert failure["replay_verified"] is False
        assert "replay mismatch" in failure["error"]
        assert not (old / name / "replay_verified.json").exists()
    else:
        evidence = read(old / name / "replay_verified.json")
        assert evidence["status"] == "verified"
        assert evidence["bitwise_full_state_proven"] is False
        refs, summary = r.replay_reference(
            r.module_for(name).ROOT / name / "training.jsonl", cp["step"])
        seen = set()
        with (old / name / "training.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                step = row["step"]
                if step in refs:
                    r.compare_replay(step, row, refs[step])
                    seen.add(step)
                if step == summary["partial_step"]:
                    r.compare_replay(step, row, summary["prefix_diagnostics"])
                    break
        assert seen == set(refs)
        assert not (s.root_for(name) / name).exists()
    print("verified:", name)

reg = m.x10.verify_registration(m.ROOT)
for path, sha in reg["artifact_sha256"].items():
    assert m.x7.digest(path) == sha, path
m.load_checkpoint()
failed = m.original.ROOT / "muon_resume"
terminal = read(failed / "receipts/muon3x_seed0/terminal.json")
assert terminal == read(failed / "markers/muon3x_seed0.FAIL")
assert terminal["exit_code"] != 0
assert terminal["status"] == "INFRA-INTERRUPTED"
assert not (failed / "muon3x_seed0/replay_verified.json").exists()
assert not read(m.OUT / "checkpoint_restored.json")["at_death_restoration_claimed"]

index = read(m.OUT / "checkpoints.json")
missing = [cp for path, cp in index.items()
           if Path(path).is_relative_to(m.OUT) and not Path(path).exists()]
print("X15 missing restart checkpoints; index hashes ONLY:")
print(json.dumps(missing, indent=2))
print("X15 pruning receipt exists:", (m.OUT / "pruning.jsonl").exists())
print("All requested immutable bindings and replay/FAIL checks passed.")
PY
```

Residual risks: concurrent external ENOSPC can defeat writes despite prechecks, as already receipted; CPU audits cannot establish historical bitwise trajectory identity or terminal scientific success.
---

## Parent attestation (2026-10-21)

Both acceptance conditions satisfied:

C1: commit c6f50c01 (21:41:50) - AMENDMENT-3-2026-10-21.md (append-only, 577 lines:
supervisor-approved evidence concession acknowledging the omitted checkpoint-deletion
receipts, full inventory of 452 missing restart checkpoints with index-recorded
sha256/step, explicit write-time-hash vs deletion-time-verification distinction,
concession scoped to the running process's remaining lifetime, prospective remedy
authorized); scripts/invertible_stage15_muon_restart.py now writes pruning.jsonl
deletion receipts (verified at line 86: append-mode receipt stream) for every future
deletion; tests extended (32 passed, 1 skipped per the worker's prelaunch receipt).
Frozen registrations and failed replay receipts untouched; running training untouched.

C2: run by the parent on lambda01 before C1 - (a) pytest across all eight stage15/16/17
test files: 80 passed, 3 skipped; (b) the reviewer's inline audit verbatim: PASS - all
registration/source/artifact/checkpoint bindings verified, replay-verified arms
(e97_w64, gdn2_w512, gdn2_w2048) re-verified diagnostic-exact with partial-row
prefixes compared, restart arms' INFRA-INTERRUPTED/FAIL receipts and replay-verified.json
absence confirmed, stage15 immutable bindings + checkpoint_restored at-death-claim
denial confirmed, missing-checkpoint inventory printed (index hashes only), no
pruning.jsonl at audit time (pre-C1, as the review found).

Provenance correction incorporated: 3562166a is a whitespace-only snapshot change;
the recovery implementation chain is cce1d03e -> 0e44cdaf / 2821b2ed -> 2d135738,
all append-only.

Verdict: BLOCKED pending C1-C2 -> ACCEPTED. X16R/17R/X15b infra-recovery admission
chain closed.
