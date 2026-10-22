## Review

**Verdict: ACCEPT-WITH-CONDITIONS.**

**Exact condition:** “The parent must run the read-only CPU audit below, retain its successful output and SHA256 manifest with this review, and resolve any assertion failure before unconditional acceptance.”

No issues found.

Paths below are relative to `experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/curriculum_extension/`.

- **Correct — endpoint:** `e97_extension_seed0/result.json:51–98,724–770` confirms T64 **[512,512,512,512,10]** and T16 **[512,512,512,512,6]**.
- **Correct — timings:** Directly inspected P2=510 at update2000 and P4 counts of 50@56000, 250@168000, 512@357500 (`validation.json:3434–3500,96114–96180,288386–288452,613634–613700`). Exhaustive **first**-crossing verification is delegated to the audit.
- **Correct — enhancement and predictions:** Teaching/retention P4 starts at **5**, while the independent T64 baseline is **9** (`registration.json:726–743`; `receipts/baseline.json:55–65`). Thus the precise improvements are T16 **5→512**, T64 **9→512**. Prediction (a)’s endpoint retention and new-acquisition criteria are satisfied. Prediction (c) registered **damage**, not enhancement; the report correctly distinguishes transient damage from recovered/enhanced fixed-final retention (`AMENDMENT.md:36–51`; `GROWTH_REPORT.md:12–28`).
- **Correct — historical contrast:** Inspected cold seed0/1 zero endpoints and seed2 **[512,512,309,1,0]**. The table compares **500K teaching + 500K extension**, not 500K total, and explicitly disclaims matched causal isolation (`GROWTH_REPORT.md:72–86`).
- **Correct — scope:** Single seed/design, P5 censoring, no fifth-position/deeper endpoint claim, schedule confounding, and both registered restart-divergence readings remain explicit (`GROWTH_REPORT.md:30–57,102–107`). The original missed teaching criterion is not retrospectively declared satisfied.
- **Correct — recorded bindings:** Configuration binds registration SHA256 `302451b5…`, teaching checkpoint `870c2dcf…`, and launch commit `10bfb5fa…`; final checkpoint receipt names `bb09d1ee…`. Worker terminal binds result SHA256 `07045c9d…`; watcher reports verified fixed-final500K (`config.json:6–13`; `result.json:883–887`; terminal receipts). Actual byte-hash verification remains conditional below.

**Review boundary:** The parent attested that the reviewed report is byte-identical to commit **e390a81a**, SHA256 `d2699f34862d9c82014f247d581ea4d1e8b5860347ec710abae7c08a74488c57`. I did not independently inspect Git history. Raw extension receipts are current-checkout, **untracked local artifacts**, not assumed committed evidence. No checkpoint replay or new training was performed.

**Merge verdict: OK with notes**, subject to the exact condition above.

## Requested read-only CPU audit

Run locally from the repository using its existing Python environment. This imports only the standard library, performs no training, and writes no repository files. Retain stdout externally with this review.

```bash
cd /home/erikg/emender
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

BASE = Path("experiments/expressivity_tasks")
X = BASE / "results_invertible_stage16_ignition_8m"
E = X / "curriculum_extension"
O = E / "e97_extension_seed0"
manifest = {}

def sha(path):
    path = Path(path)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest[str(path)] = digest
    return digest

def load(path):
    return json.loads(Path(path).read_text())

def counts(row, retention=False):
    obj = row["retention_t16"] if retention else row
    positions = obj["positions"]
    assert [p["driver_position"] for p in positions] == list(
        range(1, len(positions) + 1))
    assert all(p["count"] == 512 and 0 <= p["correct"] <= 512
               for p in positions)
    return [p["correct"] for p in positions]

assert sha(E / "GROWTH_REPORT.md") == (
    "d2699f34862d9c82014f247d581ea4d1e8b5860347ec710abae7c08a74488c57")

reg = load(E / "registration.json")
cfg = load(O / "config.json")
assert sha(E / "registration.json") == cfg["registration_sha256"]
assert cfg["registration_sha256"] == (
    "302451b536301c87b4461d268b63cdb786ede1648c4bb2878b06c801e73e68cd")
assert cfg["source_commit"] == "10bfb5fa368cb85a78c70aee6bcaad1f85937a83"
assert cfg["preserved_state_exactly_compared"] is True
assert cfg["source_checkpoint"] == reg["source_checkpoint"]
assert reg["source_checkpoint"]["sha256"] == (
    "870c2dcf5e863dcab817e3c6b0150d054e91cff8456d231378d3748676e06b1f")
assert sha(reg["source_checkpoint"]["path"]) == reg["source_checkpoint"]["sha256"]

for path, expected in reg["artifact_sha256"].items():
    assert sha(path) == expected, path
for path in [
    "scripts/invertible_stage16_extension.py",
    "tests/test_invertible_stage16_extension.py",
    str(E / "AMENDMENT.md"),
    str(X / "REGISTRATION.md"),
    str(X / "infra_restart/registration.json"),
]:
    assert sha(path) == reg["source_sha256"][path], path

teach = load(X / "infra_restart/e97_curriculum_seed0/result.json")
assert teach["status"] == "criterion_not_reached"
assert teach["total_steps"] == 500000 and teach["t64_steps"] == 0
assert counts(teach["final"]) == reg["teach_counts"]
assert teach["final"]["checkpoint"]["sha256"] == reg["source_checkpoint"]["sha256"]
admission = load(X / "infra_restart/e97_curriculum_seed0/restart_admitted.json")
assert admission["mode"] == "restart_from_checkpoint"
assert admission["restored_at_death_state"] is False
assert admission["replay_verified"] is False

result = load(O / "result.json")
terminal = load(E / "receipts/e97_extension_seed0/terminal.json")
assert terminal == load(E / "markers/e97_extension_seed0.DONE")
assert terminal["exit_code"] == 0 and terminal["error"] is None
assert sha(O / "result.json") == terminal["result_sha256"] == (
    "07045c9d6a85956926e187c4f3901f61bc774fa9c39624e67bc8b6c0beb7735a")
assert result["status"] == "complete"
assert result["teach_steps"] == result["t64_steps"] == 500000
assert result["stop_reason"] == "safety_ceiling"
watch = load(E / "receipts/watcher/terminal.json")
assert watch["exit_code"] == 0
assert watch["message"] == "verified fixed-final500K"

history = load(O / "validation.json")
steps = list(range(0, 500001, 500))
assert [r["phase_step"] for r in history] == steps
assert all(r["step"] == r["phase_step"] and r["depth"] == 64 for r in history)
assert result["final"] == history[-1]
assert load(E / "receipts/baseline.json") == history[0]
t64 = [counts(r) for r in history]
t16 = [counts(r, True) for r in history]
assert all(len(r) == 64 for r in t64)
assert all(len(r) == 16 for r in t16)
assert t64[0][:5] == [512, 512, 512, 9, 0]
assert t16[0] == reg["teach_counts"]
assert t64[-1][:5] == [512, 512, 512, 512, 10]
assert t16[-1][:5] == [512, 512, 512, 512, 6]
assert t64[4][1] == t16[4][1] == 510

def first(rows, position, threshold):
    return next((s for s, r in zip(steps, rows)
                 if r[position - 1] >= threshold), None)

assert [first(t64, 4, n) for n in (50, 250, 508, 512)] == [
    56000, 168000, 337500, 357500]
assert [first(t16, 4, n) for n in (50, 250, 508, 512)] == [
    54500, 163500, 336500, 355500]
assert [min(r[i] for r in t16) for i in range(3)] == [508, 451, 429]
assert max(r[4] for r in t64) == 17
assert max(r[4] for r in t16) < 50

material = lambda row: {i + 1 for i, n in enumerate(row) if n >= 50}
assert material(t64[0]) == {1, 2, 3}
assert set.union(*(material(r) for r in t64)) == {1, 2, 3, 4}
assert material(t64[-1]) == {1, 2, 3, 4}

# Independently reconstruct every row of both report tables.
text = (E / "GROWTH_REPORT.md").read_text()
def table(section):
    return [[cell.strip() for cell in line.strip("|").split("|")]
            for line in section.splitlines() if line.startswith("|P")]

sections = text.split("## T64 full positional receipt", 1)[1].split(
    "## retention_t16", 1)
expected64 = []
for i in range(64):
    values = [f"P{i+1}", t64[0][i], t64[-1][i],
              max(r[i] for r in t64)]
    values += [first(t64, i+1, n) for n in (50, 508, 512)]
    expected64.append(list(map(str, values)))
assert table(sections[0]) == expected64

expected16 = []
for i in range(16):
    below = next((s for s, r in zip(steps, t16) if r[i] < 508), None)
    values = [f"P{i+1}", t16[0][i], t16[-1][i],
              min(r[i] for r in t16), t16[0][i] - t16[-1][i], below]
    expected16.append(list(map(str, values)))
assert table(sections[1]) == expected16

checkpoints = load(O / "checkpoints.json")
for path in sorted((O / "T64").glob("*.pt")):
    receipt = checkpoints[str(path)]
    assert sha(path) == receipt["sha256"], str(path)
assert result["final"]["checkpoint"]["sha256"] == (
    "bb09d1eef9e3aa1d942c5efe1d36ca1bd9e6b00911d1aefd6716faedea1c4200")

for folder, prefix, maximum in [
    ("results_invertible_stage12_separation_8m/e97_bf16_seed0", [0]*5, 3),
    ("results_invertible_stage13_zerocheck_8m/e97_bf16_seed1", [0]*5, 4),
    ("results_invertible_stage13_zerocheck_8m/e97_bf16_seed2",
     [512, 512, 309, 1, 0], None),
]:
    path = BASE / folder / "validation.json"
    cold = load(path)
    assert cold[-1]["step"] == 1000000
    assert counts(cold[-1])[:5] == prefix
    if maximum is not None:
        assert max(p["correct"] for r in cold for p in r["positions"]) == maximum
    sha(path)
    del cold

# Bind remaining local raw receipts without importing training code.
for name in ("checkpoints.json", "pruning.jsonl", "result.json",
             "training.jsonl", "validation.json"):
    sha(O / name)
for path in (
    E / "receipts/e97_extension_seed0/terminal.json",
    E / "markers/e97_extension_seed0.DONE",
    E / "receipts/watcher/terminal.json",
):
    sha(path)

print("PASS: endpoints, crossings, all positional tables, contrasts and bindings")
print(json.dumps({"sha256": manifest}, indent=2, sort_keys=True))
PY
```
---

## Parent attestation (2026-10-22)

X16ER condition satisfied, with one audit-script diagnosis receipted:

1. Verbatim run of the reviewer's audit: FAILED at its line-126 table comparison.
   Diagnosis (receipted, reproducible): the audit's parser collects lines with
   startswith("|P"), which matches the report tables' HEADER row ("|Position|...")
   as a data row; the reconstructed expected list (data-only, 64 rows) then
   mismatches at index 0 against the parsed 65-row table. This is a parsing
   artifact of the audit script, not a report-data discrepancy.
2. Corrected run (the ONLY change: the parser line excludes lines starting with
   "|Position"; every other assertion byte-identical): exit 0 -
   "PASS: endpoints, crossings, all positional tables, contrasts and bindings",
   plus the full SHA256 manifest printed and retained at /tmp/x16er_audit_out.txt
   (both verbatim-failure and corrected-pass outputs retained). All checkpoint
   files re-hash to their receipts.json sha256 values; the final checkpoint
   bb09d1ee... verified; cold-contrast baselines (X12 seed0 max 3, X13 seed1
   max 4, X13 seed2 [512,512,309,1,0]) re-verified against their validation
   streams; all raw-receipt bindings hashed.

Verdict: X16ER ACCEPT-WITH-CONDITIONS -> ACCEPTED. X16E transfer finding closed.
