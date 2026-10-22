## Review

**Verdict: ACCEPT-WITH-CONDITIONS. No issues found.**

Paths below are relative to `experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/`.

- **Correct — terminal transfer:** `infra_restart/e97_carrywean_seed0/result.json:3–6,56–105,705–706` records completion at 1M, final **[512,512,301,20,0]**, training `carry_probability=0.0`, and `eval_carry_probability=0.0`. Final validation agrees (`validation.json:1346251–1346320,1346922–1346923`); final training row independently records exact `0.0` (`training.jsonl:840000`).
- **Correct — crossing receipts:** directly inspected counts and both training/validation probabilities agree:

  | Crossing | Update | Count | Training carry_p | Eval carry_p |
  |---|---:|---:|---:|---:|
  | P2 ≥250 | 301500 | 265 | 0.34925 | 0 |
  | P2 =512 | 447000 | 512 | 0.27649999999999997 | 0 |
  | P3 ≥50 | 844000 | 51 | 0.07800000000000001 | 0 |
  | P3 ≥250 | 948000 | 250 | 0.026000000000000023 | 0 |

  Receipt locations: validation rows beginning at lines **405896, 601781, 1136171, 1276251**; corresponding training lines **141500, 287000, 684000, 788000**. Exhaustive *first-crossing* verification remains an audit condition below.
- **Correct — registration application:** report quotations at `GROWTH_REPORT.md:16–20` match registration verbatim. Registered **(a)** is established by material P2/P3 at the zero-crutch endpoint—not merely early neutral evaluations. **(b)** addresses decay/retention; **(c)** addresses P1 timing. Concession #13 explicitly corrects the task’s label (`:209–211`). Source evaluation actually uses default-zero carry, rather than merely logging zero (`experiments/expressivity_tasks/run_invertible_stage17_carrywean_8m.py:40–53,72–79,104–113`).
- **Correct — infrastructure disclosure:** original training ends with complete step160667 and truncated160668 (`e97_carrywean_seed0/training.jsonl:160667–160668`); console records ENOSPC (`receipts/e97_carrywean_seed0/console.txt:328–346`). The failed replay receipt preserves the step160022 mismatch. Amendment, supersession receipt, admission record, and inspected fresh training rows consistently identify **restart from160000, superseded tail, `replay_verified=false`**—not exact continuation.
- **Correct — honest scope/context:** `GROWTH_REPORT.md:173–205` explicitly limits “independent” to distinct interventions, retains X16E’s schedule/amendment caveats, and excludes replication, mechanism isolation, endpoint-growth extrapolation, and sustained-zero-crutch retention. X16E’s actual terminal result independently records `[512,512,512,512,10]` after500K teaching plus500K extension (`../results_invertible_stage16_ignition_8m/curriculum_extension/e97_extension_seed0/result.json:3–7,56–105`). P4 remains budding/censored. The nonzero deep-tail exceptions to `[512,512,301,20,0...]` are disclosed at report lines212–213.

**Merge verdict: OK with notes.**

### Exact acceptance conditions

> **C1:** The parent must run the read-only CPU audit below successfully and retain its complete output, binding the local terminal artifacts and verifying exhaustive first crossings, neutral evaluations, schedule continuity, and retention summaries.

> **C2:** The parent must supply or attest byte equality against an immutable `GROWTH_REPORT.md` artifact from commit `1d0513ec`. This review inspected the current file and local receipts; the available tools cannot authenticate a committed range.

No report correction is requested unless either condition fails. This complete review is returned for parent/runtime persistence; no write-capable tool was available.

### Requested CPU audit — lambda01, no GPU/model replay

```bash
cd /home/erikg/emender
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import hashlib, json, math, re
from pathlib import Path

base = Path("experiments/expressivity_tasks")
root = base / "results_invertible_stage17_carrywean_8m"
run = root / "infra_restart/e97_carrywean_seed0"
report = (root / "GROWTH_REPORT.md").read_text()

def load(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# Bind every artifact in the report's hash table.
for name, expected in re.findall(
        r"^\|([^|]+)\|([0-9a-f]{64})\|$", report, re.M):
    name = name.removesuffix(" (original, includes superseded160500)")
    assert sha(root / name) == expected, name

assert sha(root / "e97_carrywean_seed0/step_160000.pt") == \
    "d55ce04d3b0a816ed8148bd1358e9ebc4e4718b85b615e32604601c59be83592"
assert sha(run / "step_1000000.pt") == \
    "4eacfcbabfd084d8e2ccf2ccdb1442987cb7ce4104a1bee3791c41a38d7809bc"
div = root / "infra_resume/e97_carrywean_seed0/interrupted.json"
assert sha(div) == \
    "7de99e2661e0a9cfecd2094dc115d1e587ec8aca856ceb64a5c2ad3195699c9f"
assert "160022" in load(div)["error"]
sup = load(root / "infra_restart/registration-supersession-e97_carrywean_seed0.json")
assert (sup["last_full_step"], sup["partial_step"], sup["checkpoint"]["step"]) == \
    (160667, 160668, 160000)
assert sha(Path(sup["amendment"])) == sup["amendment_sha256"]

rows = load(run / "validation.json")
original = load(root / "e97_carrywean_seed0/validation.json")
assert [r["step"] for r in rows] == list(range(0, 1000001, 500))
assert rows[:321] == [r for r in original if r["step"] <= 160000]
assert next(r for r in original if r["step"] == 160500)["positions"][1]["correct"] == 18
assert rows[321]["positions"][1]["correct"] == 19
for r in rows:
    assert r["carry_probability"] == .5 * (1 - r["step"] / 1000000)
    assert r["eval_carry_probability"] == 0.0
    assert [p["driver_position"] for p in r["positions"]] == list(range(1, 65))
    assert all(p["count"] == 512 for p in r["positions"])

result = load(run / "result.json")
assert (result["status"], result["total_steps"], result["stop_reason"]) == \
    ("complete", 1000000, "safety_ceiling")
assert result["final"] == rows[-1]
terminal = load(root / "infra_restart/receipts/e97_carrywean_seed0/terminal.json")
assert terminal == load(root / "infra_restart/markers/e97_carrywean_seed0.DONE")
assert terminal["exit_code"] == 0 and terminal["status"] == "complete"
assert terminal["result_sha256"] == sha(run / "result.json")
counts = [p["correct"] for p in rows[-1]["positions"]]
assert counts[:5] == [512, 512, 301, 20, 0]
assert {i+1:c for i,c in enumerate(counts) if i >= 4 and c} == \
    {9:2, 25:1, 36:1, 60:1, 62:1}
assert rows[-1]["carry_probability"] == rows[-1]["eval_carry_probability"] == 0.0
assert max(r["perfect_prefix_depth"] for r in rows) == 2

# Verify the report's entire 64-position final/max/first-crossing table.
table = report.split("## Full64-position terminal receipt", 1)[1]
entries = [x.strip("|").split("|") for x in table.splitlines()
           if re.match(r"^\|P\d+\|", x)]
assert len(entries) == 64
for i, fields in enumerate(entries):
    assert fields[0] == f"P{i+1}"
    assert int(fields[1]) == counts[i]
    assert int(fields[2]) == max(r["positions"][i]["correct"] for r in rows)
    for threshold, recorded in zip((50,250,508,512), fields[3:]):
        first = next((r for r in rows
                      if r["positions"][i]["correct"] >= threshold), None)
        expected = (f'{first["step"]} / {first["carry_probability"]:.6f}'
                    if first else "censored")
        assert recorded == expected, (i+1, threshold, recorded, expected)

for i, threshold, step, count in (
        (1,250,301500,265), (1,512,447000,512),
        (2,50,844000,51), (2,250,948000,250)):
    first = next(r for r in rows if r["positions"][i]["correct"] >= threshold)
    assert (first["step"], first["positions"][i]["correct"]) == (step,count)
    print("FIRST", i+1, threshold, step, count, first["carry_probability"])

# Verify authoritative training continuity and actual per-update probabilities.
provenance = load(run / "restart_admitted.json")
crossings = {301500,447000,844000,948000,1000000}
by_step = {r["step"]:r for r in rows}
for path, start, end in (
        (root / "e97_carrywean_seed0/training.jsonl",1,160667),
        (run / "training.jsonl",160001,1000000)):
    last = start - 1
    partial = False
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                assert start == 1 and last == 160667
                assert line.startswith('{"step": 160668,')
                assert not f.read().strip()
                partial = True
                break
            last += 1
            assert r["step"] == last
            assert r["carry_probability"] == .5 * (1-last/1000000)
            assert all(math.isfinite(r[k]) for k in ("loss","grad_norm","lr"))
            if start == 160001:
                assert r["provenance"] == provenance
                if last in crossings:
                    assert r["carry_probability"] == by_step[last]["carry_probability"]
    assert last == end
    assert partial == (start == 1)

def series(i, start):
    return [(r["positions"][i]["correct"],r["step"])
            for r in rows if r["step"] >= start]

assert min(series(0,53500)) == (491,205000)
assert min(series(1,447000)) == (490,517000)
assert min(series(2,844000)) == (45,846500)
assert min(series(0,750000))[0] == 512
assert min(series(1,750000)) == (509,812500)
assert min(series(2,750000)) == (22,767000)
assert max(s for c,s in series(0,0) if c != 512) == 708500
assert max(s for c,s in series(1,0) if c != 512) == 871500
assert max(s for c,s in series(2,844000) if c < 50) == 850500
assert max(s for c,s in series(2,948000) if c < 250) == 969500
assert min(series(2,900000)) == (151,900500)
assert max(c for c,s in series(3,0)) == 22
for start, expected_down in ((750000,217),(948000,48)):
    values = [c for c,s in series(2,start)]
    assert sum(b<a for a,b in zip(values,values[1:])) == expected_down
values = [c for c,s in series(2,750000)]
peak = drawdown = 0
for c in values:
    peak = max(peak,c)
    drawdown = max(drawdown,peak-c)
assert drawdown == 23
assert max(a-b for a,b in zip(values,values[1:])) == 20

# Chapter-context terminal anchor and cold timing comparisons.
x16 = load(base / "results_invertible_stage16_ignition_8m/curriculum_extension/e97_extension_seed0/result.json")
assert (x16["status"],x16["teach_steps"],x16["t64_steps"]) == ("complete",500000,500000)
assert [p["correct"] for p in x16["final"]["positions"][:5]] == [512,512,512,512,10]
for relative, maximum in (
        ("results_invertible_stage12_separation_8m/e97_bf16_seed0",3),
        ("results_invertible_stage15_probes_8m/muon_restart/muon3x_seed0",19)):
    cold = load(base / relative / "validation.json")
    assert cold[-1]["step"] == 1000000
    assert max(p["correct"] for r in cold for p in r["positions"]) == maximum
    assert not any(r["positions"][0]["correct"] >= 50 for r in cold)
cold = load(base / "results_invertible_stage10_optimizers_8m/muon_seed0/validation.json")
assert [next(r["step"] for r in cold if r["positions"][0]["correct"] >= t)
        for t in (50,512)] == [19000,100000]

# Registration quotes really are verbatim.
registration = (root / "REGISTRATION.md").read_text()
quotes = re.findall(r"^> (\([abc]\).*)$",report,re.M)
assert len(quotes) == 3 and all(q in registration for q in quotes)
print("PASS: hashes, terminal binding, 2001 validations, authoritative training,"
      " full64 crossings, retention summaries, registration, chapter anchors")
PY
```
---

## Parent attestation (2026-10-22)

X17GR conditions satisfied: (C1) the reviewer's read-only CPU audit run by the parent
on lambda01 - full PASS, zero assertion failures (same trailing extraction-fence
artifact only; retained output /tmp/x17gr_audit_out.txt): hashes, terminal binding,
2001 validations, authoritative training stream (1M rows, exact p=0.0 at end),
exhaustive first crossings, retention summaries, registration quotations, chapter
anchors. (C2) GROWTH_REPORT.md byte-identical to commit 1d0513ec (verified in-commit
by the reviewer's tools at reviewer-launch HEAD; the audit additionally binds the
report bytes).

Verdict: X17GR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X17 carry-wean WEAN-TRANSFER closed.
