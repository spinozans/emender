## Review

**Verdict: ACCEPT-WITH-CONDITIONS.**  
**No issues found.**

Paths below are relative to `experiments/expressivity_tasks/results_invertible_stage15_probes_8m/`.

- **Correct — terminal reading:** `muon_restart/muon3x_seed0/result.json:33–65` records completion at **1M**, validation CE **8.1597967148**, and final P1 **6/512**, with x-coordinate **512/512**. CE is **0.0619510136 below ln(3721)**—not exact marginality or zero learning.
- **Correct — registered interpretation:** `REGISTRATION.md:63–78` defines escape as **any position ≥50/512**. The addendum quotes both predictions verbatim and distinguishes **(a) not met** from **(b)’s exact-marginal premise not established** (`muon_restart/X15G-growth-report-addendum.md:23–46`). The terminal audit reports P1 maximum **19 at 41K and 84K**, P2+ maximum **4**, and no crossings.
- **Correct — boundary-spanning plateau:** raw `muon_restart/muon3x_seed0/validation.json` confirms P1 **8@433K, 10@500K, 10@700K, 8@900K, 6@1M** at lines **619193, 714741, 999945, 1285145, 1427745**. The early maximum **19@41K** is recorded at line **58629**. This supports a descriptive plateau across the boundary, **not** the lost uninterrupted counterfactual.
- **Correct — infrastructure scope:** amendment-1’s terminal receipt remains **INFRA-INTERRUPTED**, exit **1**, with the exact **433044** discrepancy approximately **5.23e-11** (`muon_resume/receipts/muon3x_seed0/terminal.json:2–10`). Amendment-2 authorizes checkpoint restart, not exact historical replay. `muon_restart/muon3x_seed0/checkpoint_restored.json:2–8` explicitly records **`at_death_restoration_claimed=false`**.
- **Correct — synthesis:** the report says **2/3 AdamW seeds dead**, not all three; seed2 escaped, ending two-perfect with P3 **309**. Muon@3x adds one no-escape seed. SF evidence is scoped to **T16/T128**, explicitly **untested cold at T64**, with **concession #12** recorded (`muon_restart/X15G-growth-report-addendum.md:112–144`). This is not a matched three-optimizer T64 comparison.
- **Correct — amendment-3 status:** its pruning concession covers the already-running process’s remaining lifetime. It **does not establish historical compliance**, deletion-time hashes, timestamps, or deleted-byte identity. The prospective remedy applies only to separately authorized future launches (`muon_restart/AMENDMENT-3-2026-10-21.md:3–63`). The terminal addendum retains that limitation.

**Exact condition text:**

> Acceptance becomes unconditional only after the parent retains a successful independent read-only CPU audit of the complete validation/signature and finite consecutive update streams, registered cosine, admission-source/dependency/pool bindings, terminal receipts and retained checkpoint hashes, and confirms that the reviewed report, amendments and terminal-audit receipt match commit a68bbd34. Any mismatch requires re-review. Amendment-3’s pruning evidence concession and amendment-2’s denial of at-death restoration remain explicit limitations.

**Merge verdict: OK with notes**, subject to that validation condition. No corrective document edit is requested.

### Requested parent audit

Run on **lambda01**, from `/home/erikg/emender`. This reads artifacts only; it does not import the trainer, launch training, or regenerate reports.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import hashlib, json, math, subprocess
from pathlib import Path
import numpy as np

B = Path("experiments/expressivity_tasks/results_invertible_stage15_probes_8m")
R = B / "muon_restart"
O = R / "muon3x_seed0"
def J(p): return json.loads(Path(p).read_text())
def H(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def blob(commit, p):
    return subprocess.check_output(["git", "show", f"{commit}:{p}"])

for p in [
    R/"X15G-growth-report-addendum.md",
    R/"AMENDMENT-2-2026-10-21.md",
    R/"AMENDMENT-3-2026-10-21.md",
    R/"receipts/terminal_audit.json",
]:
    assert blob("a68bbd34", p) == p.read_bytes(), ("commit mismatch", str(p))

a = J(R/"receipts/terminal_audit.json")
reg = J(R/"registration.json")
assert len(reg["source_sha256"]) == 186
for p, sha in reg["source_sha256"].items():
    assert hashlib.sha256(blob(a["admission_commit"], p)).hexdigest() == sha, p
assert len(reg["dependency_sha256"]) == 388
assert len(reg["artifact_sha256"]) == 5
for key in ("dependency_sha256", "artifact_sha256"):
    for p, sha in reg[key].items():
        assert H(p) == sha, p
with np.load(B/"muon3x_seed0/frozen_validation.npz", allow_pickle=False) as z:
    hashes = [hashlib.sha256(z[f"T64_{i}"].tobytes()).hexdigest()
              for i in range(3)]
assert hashes == reg["pool_array_sha256"]["0"] == a["pool_array_sha256_verified"]

v = J(O/"validation.json")
assert [r["step"] for r in v] == list(range(0, 1000001, 500))
assert v[:867] == J(B/"muon3x_seed0/validation.json")
restored = J(O/"checkpoint_restored.json")
assert restored == a["restoration_receipt"]
assert restored["model_optimizer_rng_exact"] is True
assert restored["at_death_restoration_claimed"] is False
provenance = restored["provenance"]
assert all(r["provenance"] == provenance for r in v[867:])
for r in v:
    assert [p["driver_position"] for p in r["positions"]] == list(range(1,65))
    assert all(p["count"] == 512 and math.isfinite(p["loss"])
               for p in r["positions"])
    assert math.isfinite(r["aggregate"]["loss"])
counts = [[p["correct"] for p in r["positions"]] for r in v]
assert max(c[0] for c in counts) == 19
assert [r["step"] for r,c in zip(v,counts) if c[0] == 19] == [41000,84000]
assert max(max(c[1:]) for c in counts) == 4
assert not any(n >= 50 for c in counts for n in c)
by_step = {r["step"]: r for r in v}
assert [by_step[s]["positions"][0]["correct"]
        for s in (433000,500000,700000,900000,1000000)] == [8,10,10,8,6]

sig = J(R/"marginal_signature.json")
expected = [{
    "step": r["step"], "ce": r["aggregate"]["loss"],
    "ce_minus_ln3721": r["aggregate"]["loss"] - math.log(3721),
    "max_position_correct": max(p["correct"] for p in r["positions"]),
    "material_positions": [p["driver_position"] for p in r["positions"]
                           if p["correct"] >= 50],
    "position_correct": [p["correct"] for p in r["positions"]],
    "position_ce": [p["loss"] for p in r["positions"]],
} for r in v]
assert sig["reference_ce"] == math.log(3721) and sig["rows"] == expected
assert H(O/"validation.json") == a["validation_sha256"]
assert H(R/"marginal_signature.json") == a["signature_sha256"]

for stream in a["streams"]:
    assert H(stream["path"]) == stream["sha256"]
    want, seen, boundary = stream["start"], 0, 0
    with Path(stream["path"]).open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "restart_from_checkpoint":
                boundary += 1
                assert row["step"] == 433000
                assert row["provenance"] == provenance
                continue
            step = row["step"]
            assert step == want
            assert all(math.isfinite(row[k])
                       for k in ("loss","grad_norm","lr","optimizer_seconds"))
            lr = (.003 * step / 5000 if step <= 5000 else
                  .00003 + .5 * (.003-.00003) *
                  (1 + math.cos(math.pi*(step-5000)/(1000000-5000))))
            assert row["lr"] == lr, step
            if stream["start"] == 433001:
                assert row["provenance"] == provenance
            want += 1
            seen += 1
            if step == stream["end"]:
                if stream["start"] == 433001:
                    assert not f.read().strip(), "unexpected restart tail"
                break
    assert seen == stream["consecutive_finite_updates"]
    assert want == stream["end"] + 1
    assert boundary == (1 if stream["start"] == 433001 else 0)

result = J(O/"result.json")
terminal = J(R/"receipts/muon3x_seed0/terminal.json")
assert terminal == J(R/"markers/muon3x_seed0.DONE")
assert terminal["exit_code"] == 0 and terminal["status"] == "complete"
assert H(O/"result.json") == terminal["result_sha256"] == a["terminal_result_sha256"]
assert result["status"] == "complete" and result["total_steps"] == 1000000
assert result["stop_reason"] == "safety_ceiling" and result["final"] == v[-1]
assert result["final"]["perfect_prefix_depth"] == 0
assert result["final"]["aggregate"]["loss"] == 8.159796714782715
index = J(O/"checkpoints.json")
for key in ("final_checkpoint", "best_checkpoint"):
    cp = a[key]
    assert H(cp["path"]) == cp["sha256"]
    assert index[cp["path"]] == cp
assert result["final"]["checkpoint"] == a["final_checkpoint"]
assert result["best"]["checkpoint"] == a["best_checkpoint"]
assert H(reg["resume"]["checkpoint"]) == reg["resume"]["sha256"]

print("PASS: commit bindings, sources/dependencies/pool, streams/cosine, "
      "all64 signature, terminal receipts and retained checkpoints")
print("P1 final=6; max=19@41000/84000; P2+ max=4; crossings=none")
print("Final CE-ln3721 =", result["final"]["aggregate"]["loss"]-math.log(3721))
PY
```

Do **not** use the watcher’s report-generation mode for this audit: it writes the addendum/signature.

Review returned here for persistence; no repository files were edited. I inspected current artifacts, not the committed range itself, and did not execute CPU commands.
---

## Parent attestation (2026-10-22)

X15GR condition satisfied: the reviewer's read-only CPU audit run VERBATIM by the
parent on lambda01, exit 0 - "PASS: commit bindings, sources/dependencies/pool,
streams/cosine, all64 signature, terminal receipts and retained checkpoints";
terminal readings re-derived (P1 final=6; max=19@41000/84000; P2+ max=4;
crossings=none; CE-ln(3721)=-0.061951013563907864). Output retained at
/tmp/x15gr_audit_out.txt. Report/amendments/terminal-audit match commit a68bbd34
(verified inside the audit). Amendment-3's pruning concession and amendment-2's
at-death-restoration denial stand as explicit limitations, unaltered.

Verdict: X15GR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X15G Muon-T64 NO-ESCAPE closed.
