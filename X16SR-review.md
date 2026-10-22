## Review

**Verdict: ACCEPT-WITH-CONDITIONS.**

No issues found.

- **Correct — terminal readings:** All four terminal receipts report `exit_code=0`, `status=complete`, and hashes matching the report’s hash table. All four `result.json` files report 500000 updates, `safety_ceiling`, and final step 500000. Final losses match **7.971306, 7.968394, 8.221836, 8.234554** respectively.
- **Correct — interpretation:** `GROWTH_REPORT.md:4–26,38–58,280–293` explicitly distinguishes the E97 **(1/32,1/16]** and GDN2 **(1/512,1/136]** observational brackets, historical-dose **4.25–8.5×** comparison, and conditional approximately **4–32×** threshold-ratio bound. Concession #14 is explicit. Predictions (a)/(b) reproduce the registration; no interior-point result, monotonicity, or multi-seed generality is asserted.
- **Correct — provenance:** `GROWTH_REPORT.md:113–150` matches the three `replay_verified.json` receipts and E97 w32’s `restart_admitted.json:2–9`. The latter expressly denies replay verification and at-death restoration.
- **Correct — positional disclosure:** `GROWTH_REPORT.md:169–256` contains all **64 positional rows**, separately labels P1 versus composition, and right-censors absent crossings. I compared their final counts/CEs against the result receipts. Exhaustive maxima, schedule, historical-onset, and cryptographic recomputation remain subject to the CPU audit below.
- **Merge verdict:** **OK with notes**, conditional on the following evidence gate.

### Exact condition text

> Acceptance becomes unconditional only after the parent runs the read-only CPU audit below successfully, retains its stdout with this review, and verifies that the reviewed GROWTH_REPORT.md is byte-identical to commit 1b917ce0. Any failed assertion requires resolution and re-review before acceptance.

The available diff tool establishes no working-tree changes for this report against reviewer-launch HEAD `1d0513ec20ca`; it does **not** independently inspect commit `1b917ce0`.

### Requested read-only CPU audit

Run from the repository root **on lambda01**, using its existing Python 3.12 venv. No training, GPU, or checkpoint loading is involved.

```bash
.venv/bin/python - <<'PY'
import hashlib, json, re, subprocess
from pathlib import Path

base = Path("experiments/expressivity_tasks")
root = base / "results_invertible_stage16_ignition_8m"
report = root / "GROWTH_REPORT.md"
raw = report.read_bytes()
assert raw == subprocess.check_output(
    ["git", "show", f"1b917ce0:{report.as_posix()}"]
), "Report differs from commit 1b917ce0"
text = raw.decode()
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

quoted = text.split("Measurement1 predictions **verbatim**:", 1)[1]
quoted = quoted.split("\n\nApplied as registered:", 1)[0]
quoted = " ".join(x[2:] for x in quoted.splitlines() if x.startswith("> "))
registered = (root / "REGISTRATION.md").read_text()
registered = registered.split("Registered predictions:\n", 1)[1].split("\n(c)", 1)[0]
assert " ".join(quoted.split()) == " ".join(registered.split())

rows = re.findall(
    r'^\|P(\d+) (non-composition|composition)\|(\d+)\|(\d+)\|'
    r'([^|]+)\|([^|]+)\|([^|]+)\|([0-9.]+)\|$', text, re.M)
assert len(rows) == 64
hashes = {
    p: (r, v) for p, r, v in re.findall(
        r'^\|`(infra_(?:resume|restart)/[^`]+)`\|`([0-9a-f]{64})`'
        r'\|`([0-9a-f]{64})`\|$', text, re.M)
}
arms = [
    ("infra_resume", "e97_w64_seed0", 11, 20, "7.971306"),
    ("infra_restart", "e97_w32_seed0", 6, 19, "7.968394"),
    ("infra_resume", "gdn2_w512_seed0", 0, 2, "8.221836"),
    ("infra_resume", "gdn2_w2048_seed0", 0, 2, "8.234554"),
]
assert len(hashes) == 4
for a, (ns, name, final_p1, max_p1, loss) in enumerate(arms):
    path = root / ns / name
    result = load(path / "result.json")
    series = load(path / "validation.json")
    terminal = load(root / ns / "receipts" / name / "terminal.json")
    assert terminal["exit_code"] == 0 and terminal["status"] == "complete"
    assert terminal["result_sha256"] == sha(path / "result.json")
    assert hashes[f"{ns}/{name}"] == (
        sha(path / "result.json"), sha(path / "validation.json"))
    assert result["status"] == "complete"
    assert result["total_steps"] == 500000
    assert result["stop_reason"] == "safety_ceiling"
    assert result["final"]["step"] == 500000
    assert [v["step"] for v in series] == list(range(0, 500001, 500))
    assert result["final"]["positions"] == series[-1]["positions"]
    assert f'{result["final"]["aggregate"]["loss"]:.6f}' == loss
    for v in series:
        assert [p["driver_position"] for p in v["positions"]] == list(range(1, 17))
        assert all(p["count"] == 512 for p in v["positions"])
    maxima = [max(v["positions"][i]["correct"] for v in series)
              for i in range(16)]
    assert maxima[0] == max_p1 and max(maxima[1:]) == 3
    assert series[-1]["positions"][0]["correct"] == final_p1
    for i, row in enumerate(rows[a*16:(a+1)*16]):
        pos, role, final, maximum, *rest = row
        assert int(pos) == i+1
        assert role == ("non-composition" if i == 0 else "composition")
        assert int(final) == series[-1]["positions"][i]["correct"]
        assert int(maximum) == maxima[i]
        for threshold, displayed in zip((50, 508, 512), rest[:3]):
            crossing = next((v["step"] for v in series
                             if v["positions"][i]["correct"] >= threshold), None)
            assert crossing is None and displayed == "censored@500000"
        assert rest[3] == f'{series[-1]["positions"][i]["loss"]:.6f}'
    print(name, "complete500K", "P1max", maxima[0],
          "P2+max", max(maxima[1:]), "CE", loss, "16 rows PASS")

for name, checkpoint, updates in (
    ("e97_w64_seed0", 244000, 106),
    ("gdn2_w512_seed0", 129000, 920),
    ("gdn2_w2048_seed0", 130000, 148),
):
    receipt = load(root / "infra_resume" / name / "replay_verified.json")
    assert receipt["status"] == "verified"
    assert receipt["checkpoint"]["step"] == checkpoint
    assert receipt["completed_replay_updates"] == updates
    assert receipt["last_full_step"] == checkpoint + updates
    assert receipt["partial_step"] == checkpoint + updates + 1
    assert receipt["prefix_diagnostics"] == {}
    assert receipt["bitwise_full_state_proven"] is False

restart = load(root / "infra_restart/e97_w32_seed0/restart_admitted.json")
assert restart["mode"] == "restart_from_checkpoint"
assert restart["checkpoint_step"] == 245000
assert restart["historical_tail_superseded_from"] == 245001
assert restart["historical_last_full_step"] == 245475
assert restart["historical_partial_death_step"] == 245476
assert restart["replay_verified"] is False
assert restart["restored_at_death_state"] is False

historical = [
    ("results_invertible_stage14_nofreebie_8m/e97_seed0", None, None),
    ("results_invertible_stage14_nofreebie_8m/gdn2_seed0", None, None),
    ("results_invertible_stage9_anneal_8m/e97_bf16_seed0", 47000, [512]*3),
    ("results_invertible_stage10_optimizers_8m/adamw_seed0", 24000, [512]*4),
    ("results_invertible_stage8_paydepth_8m/gdn2_bf16_seed0", 139500, [505,511]),
]
for relative, expected_first, expected_final in historical:
    path = base / relative
    series = load(path / "validation.json")
    result = load(path / "result.json")
    assert result["status"] == "complete" and result["total_steps"] == 500000
    assert series[-1]["step"] == result["final"]["step"] == 500000
    first = min((v["step"] for v in series
                 if any(p["correct"] >= 50 for p in v["positions"])),
                default=None)
    assert first == expected_first, (relative, first)
    if expected_final is not None:
        assert [p["correct"] for p in result["final"]["positions"]
                ][:len(expected_final)] == expected_final
    print(relative, "first50", first, "PASS")

print("PASS: commit binding, registered quotations, hashes, four terminals, "
      "64 rows, schedules, historical endpoints, and admission split")
PY
```

Residual scientific limits remain correctly disclosed: seed0 only, fixed 500K ceiling, unmatched historical X8 recipe, and no bitwise full-state replay proof.

Review returned for persistence; no repository files were modified.
---

## Parent attestation (2026-10-22)

X16SR condition satisfied: the reviewer's read-only CPU audit run by the parent on
lambda01 - full PASS with zero assertion failures (the shell's trailing exit=2 is the
awk-extraction fence artifact; the Python heredoc itself completed; zero
Traceback/AssertionError lines in the retained output /tmp/x16sr_audit_out.txt).
Verified: GROWTH_REPORT.md byte-identical to commit 1b917ce0; registered predictions
quoted verbatim; all four terminal result/validation/hash bindings re-derived
(e97_w64 P1max 20 CE 7.971306; e97_w32 P1max 19 CE 7.968394; gdn2_w512 P1max 2 CE
8.221836; gdn2_w2048 P1max 2 CE 8.234554; all 500K complete); 64 positional rows
reconstructed; schedules and historical endpoints re-verified (X14 both-arms no
crossings; X9-anneal first50@47K; X10-adamw first50@24K; X8-GDN2 first50@139.5K);
admission split (3 replay-verified, 1 restart) confirmed.

Verdict: X16SR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X16 ignition-sweep closed.
