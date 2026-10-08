#!/usr/bin/env python3
"""CPU unit validation for the e97-rl-loop-v1 bank (tranche 3) — run BEFORE
any GPU launch.  Everything here is fail-closed: any assertion failure is a
build defect.  Covers:

  1. GLOBAL POOL lease semantics (no SQLite anywhere): exactly-one-winner
     atomic claims under REAL process concurrency; heartbeat extends a live
     lease; an expired/crashed claim is swept back to pending and becomes
     claimable again; retirement lands in done/ with an outcome block.
  2. DILOCO MERGE math on synthetic checkpoints: the weighted-soup-by-
     pass-rate reproduces exact fp32 reference math in the checkpoint
     dtype; exp_avg_sq + scalar clocks come from the winner; zero-signal
     merges are SKIPPED with a receipt and NOTHING published; merge
     receipts are digest-linked (hash chain) and the pointer advances only
     from the readable, sha-verified merged file (ADR-003 R07).
  3. K-TRIGGER snapshot semantics: a lane's updates since the last merge
     (updates_total - coordinator snapshot) reach K exactly once per
     window.
  4. REAL LAKE admission path (CPU): the pinned validator program is
     recovered byte-exact (344a1209…), the task-lake validation layer
     re-verifies the admitted lake, and a pool refresh freezes the sealed
     bundles with round-scoped ids bound to the pinned program bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_bank import (MERGE_CURRENT_SCHEMA, bank_paths, claim_pool_task,
                     ensure_bank_layout, ensure_pinned_validator,
                     heartbeat_claim, init_lane_state, lane_paths,
                     merge_stream_tail, pool_status, read_merge_current,
                     refresh_pool, retire_task, sweep_stale_claims,
                     write_lane_state, PINNED_VALIDATOR_SHA256)
from rl_bank_merge import publish_merge_current, run_merge
from rl_bank_coordinator import trigger_check

REPO_ROOT = Path("/home/erikg/emender")
LAKE = Path("/mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-admitted")

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SystemExit(f"UNIT TEST FAILED: {name} {detail}")
    PASSED.append(name)
    print(f"PASS {name}" + (f" ({detail})" if detail else ""), flush=True)


# --------------------------------------------------------------------- 1
def _claim_worker(bank_root: str, lane: int, out: str) -> None:
    bank = bank_paths(bank_root)
    task = claim_pool_task(bank, lane, ttl_seconds=60)
    if task is None:
        (Path(out)).write_text("none")
        return
    (Path(out)).write_text(task["task_id"])


def test_pool(scratch: Path) -> None:
    bank = bank_paths(scratch)
    ensure_bank_layout(bank)
    for index in range(3):
        task = {
            "schema": "emender-rl-loop-seed-task-v1",
            "task_id": f"unit-pool-{index}", "attempts": 0, "round": 1,
            "body": {"template": "unit", "prompt": "x"},
            "task_sha256": f"{index:064d}", "generator": "unit test",
            "authoring": "unit test", "source_commit": "0" * 40,
        }
        from rl_bank import freeze_pool_task

        freeze_pool_task(bank, task)
    outs = [scratch / f"claim-{index}.out" for index in range(4)]
    procs = [subprocess.Popen(
        [sys.executable, __file__, "claim-worker", str(scratch),
         str(lane), str(out)]) for lane, out in enumerate(outs)]
    for proc in procs:
        proc.wait()
    claimed = [out.read_text() for out in outs if out.read_text() != "none"]
    check("pool: 3 tasks claimed by 4 racing lanes, no double claim",
          sorted(claimed) == ["unit-pool-0", "unit-pool-1", "unit-pool-2"],
          f"claims={sorted(claimed)}")
    check("pool: pending drained", pool_status(bank)["pending"] == 0)

    # heartbeat extends the live lease (use the claim's ACTUAL winner lane
    # — the race above does not guarantee which lane holds which task)
    task = json.loads((bank["pool_claims"] / "unit-pool-0.claim").read_text())
    winner_lane = int(task["claim"]["lane"])
    check("pool: heartbeat extends the lease",
          heartbeat_claim(bank, task, lane=winner_lane, ttl_seconds=60))
    status = pool_status(bank)
    check("pool: live claims counted", status["claims_live"] == 3)

    # expired claim swept + re-claimable by another lane
    task_path = bank["pool_claims"] / "unit-pool-0.claim"
    stale = json.loads(task_path.read_text())
    stale["claim"]["deadline_unix"] = time.time() - 1
    temporary = task_path.with_name(".stale.tmp")
    temporary.write_text(json.dumps(stale, sort_keys=True, indent=1))
    os.replace(temporary, task_path)
    moved = sweep_stale_claims(bank)
    check("pool: stale claim requeued", moved == 1
          and (bank["pool_pending"] / "unit-pool-0.json").is_file())
    re_claimed = claim_pool_task(bank, 7, ttl_seconds=60)
    check("pool: requeued task re-claimable by another lane",
          re_claimed is not None and re_claimed["task_id"] == "unit-pool-0")
    retire_task(bank, re_claimed, outcome={"lane": 7, "cycle": 1,
                                           "stage": "policy"})
    retired = json.loads((bank["pool_done"] / "unit-pool-0.json").read_text())
    check("pool: retirement outcome evidence",
          retired["retired"]["outcome"]["lane"] == 7
          and retired["attempts"] == 1)


# --------------------------------------------------------------------- 2
def _synthetic_checkpoint(path: Path, value: float, tag: str) -> None:
    payload = {
        "schema": "emender-e97-4b-pi-masked-sft-v1",
        "model_state_dict": {"w": torch.tensor([value, value * 2],
                                                dtype=torch.bfloat16)},
        "optimizer_state_dict": {
            "param_groups": [{"lr": 1e-05, "weight_sum": 1.0, "k": 5,
                              "params": [0]}],
            "state": {0: {"z": torch.tensor([value + 100],
                                             dtype=torch.bfloat16),
                          "exp_avg_sq": torch.tensor([value],
                                                      dtype=torch.bfloat16)}},
        },
        "weight_mode": "saved-eval-x",
        "trainer": f"synthetic-{tag}",
        "sft_updates": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_merge(scratch: Path) -> None:
    bank = bank_paths(scratch)
    ensure_bank_layout(bank)
    lanes_root = scratch / "merge-lanes"
    paths = []
    for lane, value in enumerate((3.0, 5.0, 11.0)):
        path = lanes_root / f"lane-{lane:02d}" / "checkpoint.pt"
        _synthetic_checkpoint(path, value, f"lane{lane}")
        paths.append(path)

    def state(lane: int, attempts: int, passes: int) -> dict:
        return {"lane": lane, "lineage_path": str(paths[lane]),
                "lineage_sha256": _sha(paths[lane]),
                "window_attempts": attempts, "window_passes": passes,
                "window_receipts": passes, "updates_total": 32}

    # weights: lane0 2/4 = .5, lane1 1/2 = .5, lane2 0/1 = 0 -> soup of
    # lanes 0+1 only, normalized (.5, .5); lane2 contributes nothing
    states = [state(0, 4, 2), state(1, 2, 1), state(2, 1, 0)]
    receipt = run_merge(bank, merge_index=1, k=32, trigger_lane=0,
                        trigger_updates=32, lane_states=states)
    check("merge: published", receipt["status"] == "published")
    merged_path = Path(receipt["merged_checkpoint_path"])
    check("merge: atomic publish exists + sha binds",
          merged_path.is_file()
          and _sha(merged_path) == receipt["merged_checkpoint_sha256"])
    merged = torch.load(str(merged_path), map_location="cpu")
    expected_w = (0.5 * torch.tensor([3.0, 6.0], dtype=torch.float32)
                  + 0.5 * torch.tensor([5.0, 10.0], dtype=torch.float32)
                  ).to(torch.bfloat16)
    check("merge: weighted-soup math exact (x)",
          torch.equal(merged["model_state_dict"]["w"], expected_w),
          f"got={merged['model_state_dict']['w'].tolist()}")
    expected_z = (0.5 * torch.tensor([103.0], dtype=torch.float32)
                  + 0.5 * torch.tensor([105.0], dtype=torch.float32)
                  ).to(torch.bfloat16)
    check("merge: weighted-soup math exact (z)",
          torch.equal(merged["optimizer_state_dict"]["state"][0]["z"],
                      expected_z))
    check("merge: exp_avg_sq from the winner (lane0, rank rule)",
          torch.equal(merged["optimizer_state_dict"]["state"][0]["exp_avg_sq"],
                      torch.tensor([3.0], dtype=torch.bfloat16)))
    check("merge: scalar clocks from the winner",
          merged["optimizer_state_dict"]["param_groups"][0]["k"] == 5)
    check("merge: receipt digest-linked chain head",
          merge_stream_tail(bank) == receipt["receipt_sha256"])

    current = publish_merge_current(bank, receipt, epoch=1)
    check("merge: pointer advances from the verified merged file",
          current["schema"] == MERGE_CURRENT_SCHEMA
          and read_merge_current(bank)["merged_checkpoint_sha256"]
          == receipt["merged_checkpoint_sha256"])

    # zero-signal: NO lane has a pass -> SKIP, nothing published
    zero_states = [state(0, 3, 0), state(1, 2, 0)]
    skip = run_merge(bank, merge_index=2, k=32, trigger_lane=1,
                     trigger_updates=40, lane_states=zero_states)
    check("merge: zero-signal merge SKIPPED with receipt",
          skip["status"] == "skipped" and "skip_reason" in skip)
    check("merge: skip published nothing",
          "merged_checkpoint_path" not in skip)
    check("merge: skip receipt chains onto the first",
          skip["prev_receipt_sha256"] == receipt["receipt_sha256"]
          and merge_stream_tail(bank) == skip["receipt_sha256"])


# --------------------------------------------------------------------- 3
def test_trigger() -> None:
    states = [{"lane": lane, "updates_total": updates}
              for lane, updates in enumerate((10, 32, 5))]
    check("trigger: lane hitting K fires",
          trigger_check(states, {}, 32) == (1, 32))
    states = [{"lane": lane, "updates_total": updates}
              for lane, updates in enumerate((10, 40, 5))]
    check("trigger: snapshot window counts only post-merge updates",
          trigger_check(states, {str(lane): 10 for lane in range(3)}, 32)
          is None)
    states = [{"lane": lane, "updates_total": updates}
              for lane, updates in enumerate((43, 40, 5))]
    check("trigger: lane exceeding snapshot by K fires",
          trigger_check(states, {str(lane): 10 for lane in range(3)}, 32)
          == (0, 33))


# --------------------------------------------------------------------- 4
def test_lake(scratch: Path) -> None:
    bank = bank_paths(scratch)
    ensure_bank_layout(bank)
    program = ensure_pinned_validator(bank)
    check("lake: pinned validator bytes bound to the lake pin",
          _sha(program) == PINNED_VALIDATOR_SHA256, _sha(program))
    record = refresh_pool(bank, lake=LAKE, round_number=1, program_path=program)
    check("lake: 2 train bundles admitted through the validation layer",
          len(record["frozen"]) == 2,
          f"froze={[f['task_id'] for f in record['frozen']]}")
    task = json.loads((bank["pool_pending"] /
                       f"{record['frozen'][0]['task_id']}.json").read_text())
    binding = task["body"]["task_lake"]
    check("lake: task binds the pinned program bytes",
          binding["validator"]["program_sha256"] == PINNED_VALIDATOR_SHA256)
    splits = sorted(f["split"] for f in record["frozen"])
    check("lake: sealed train/development split isolation present",
          splits == ["train", "train"]
          and len(record["skipped"]) == 2
          and all(f["split"] == "development" and f["reason"] == "non-training-split"
                  for f in record["skipped"]))
    eligible = [f["receipt_eligible"] for f in record["frozen"]
                if f["split"] == "train"]
    check("lake: only train split receipt-eligible", all(eligible))
    # lane init against a synthetic seed
    seed = scratch / "seed.pt"
    _synthetic_checkpoint(seed, 1.0, "seed")
    state = init_lane_state(bank, 0, checkpoint=seed,
                            checkpoint_sha256=_sha(seed), note="unit")
    check("lane: state initialized", state["lane"] == 0
          and state["lineage_sha256"] == _sha(seed))
    lane = lane_paths(bank, 0)
    state["updates_total"] = 5
    write_lane_state(lane["root"], state)
    from rl_bank import read_lane_state

    check("lane: state round-trips", read_lane_state(lane["root"])["updates_total"] == 5)


def test_adoption(scratch: Path) -> None:
    """Merge adoption resets the per-window counters (soup weights are the
    pass rate SINCE THE LAST MERGE — supervisor-approved semantics fix;
    updates_total stays monotonic for the K trigger)."""
    from rl_bank import publish_merge_current as _publish
    from rl_bank_lane import adopt_merge_current

    bank = bank_paths(scratch)
    ensure_bank_layout(bank)
    seed = scratch / "adopt-seed.pt"
    _synthetic_checkpoint(seed, 1.0, "seed")
    state = init_lane_state(bank, 0, checkpoint=seed,
                            checkpoint_sha256=_sha(seed), note="unit")
    state.update(window_attempts=7, window_passes=3, window_receipts=4,
                 updates_total=9)
    write_lane_state(bank["lanes"] / "lane-00", state)
    check("adoption: no pointer, no adoption",
          adopt_merge_current(bank, state) is None
          and state["window_attempts"] == 7)

    merged = scratch / "adopt-merged.pt"
    _synthetic_checkpoint(merged, 2.0, "merged")
    current = {"schema": MERGE_CURRENT_SCHEMA, "epoch": 1,
               "merge_receipt_sha256": "r" * 64,
               "merged_checkpoint_path": str(merged),
               "merged_checkpoint_sha256": _sha(merged),
               "published_unix": time.time()}
    _publish(bank, current)
    adopted = adopt_merge_current(bank, state)
    check("adoption: pointer adopted (sha-verified lineage)",
          adopted is not None and state["merge_epoch"] == 1
          and state["lineage_sha256"] == _sha(merged))
    check("adoption: window counters reset (per-merge window)",
          (state["window_attempts"], state["window_passes"],
           state["window_receipts"]) == (0, 0, 0))
    check("adoption: updates_total preserved (monotonic K counter)",
          state["updates_total"] == 9)
    check("adoption: same epoch not re-adopted",
          adopt_merge_current(bank, state) is None)


def test_reclaim(scratch: Path) -> None:
    """RECLAIM-LOG discipline: superseded lineage checkpoints are cache,
    not evidence.  Happy path deletes + logs sha/reason; every refusal path
    fails closed (merged, cross-lane current, outside the lane's training
    tree, identity drift, grace window)."""
    from rl_bank import (append_reclaim_record, disk_free_bytes, disk_low,
                         reclaim_superseded_lineages)

    bank = bank_paths(scratch)
    ensure_bank_layout(bank)
    training = bank["lanes"] / "lane-00" / "training"
    training.mkdir(parents=True, exist_ok=True)

    def candidate(name: str, value: float = 1.0) -> dict:
        path = training / f"{name}.pt"
        _synthetic_checkpoint(path, value, name)
        return {"path": str(path), "sha256": _sha(path),
                "superseded_unix": time.time() - 3600.0}

    # happy path: grace-elapsed own training artifact -> deleted + logged
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [candidate("gone")]}
    reclaimed = reclaim_superseded_lineages(bank, state)
    log = [json.loads(line) for line in
           Path(bank["reclaim_log"]).read_text().splitlines() if line.strip()]
    check("reclaim: deletes grace-elapsed own checkpoint",
          len(reclaimed) == 1 and not Path(reclaimed[0]["path"]).exists())
    check("reclaim: logs sha + reason (RECLAIM-LOG)",
          len(log) == 1 and log[0]["sha256"] == reclaimed[0]["sha256"]
          and "superseded lineage checkpoint" in log[0]["reason"])
    check("reclaim: ledger emptied on reclaim",
          state["superseded_lineages"] == [])

    # merged checkpoints are kept forever (merge-current or under merges/)
    merged_file = bank["merges"] / "merge-0001" / "checkpoint.pt"
    _synthetic_checkpoint(merged_file, 2.0, "merged")
    entry = candidate("merged-ref")
    entry["path"] = str(merged_file)
    entry["sha256"] = _sha(merged_file)
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [entry]}
    check("reclaim: refuses merged checkpoint (kept forever)",
          reclaim_superseded_lineages(bank, state) == []
          and merged_file.is_file()
          and state["superseded_lineages"] == [])

    # another lane's CURRENT lineage is never reclaimable
    other = candidate("other-lane")
    other_lane_state = {"lane": 1, "lineage_path": other["path"],
                        "lineage_sha256": other["sha256"]}
    write_lane_state(bank["lanes"] / "lane-01",
                    {"schema": "emender-rl-loop-bank-lane-state-v1",
                     **other_lane_state, "superseded_lineages": []})
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [other]}
    check("reclaim: refuses cross-lane current lineage",
          reclaim_superseded_lineages(bank, state) == []
          and Path(other["path"]).is_file()
          and state["superseded_lineages"] == [other])

    # outside this lane's training tree (the shared v1 seed precedent)
    seed = scratch / "shared-seed.pt"
    _synthetic_checkpoint(seed, 3.0, "seed")
    outside = {"path": str(seed), "sha256": _sha(seed),
               "superseded_unix": time.time() - 3600.0}
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [outside]}
    check("reclaim: refuses outside-lane-tree artifacts",
          reclaim_superseded_lineages(bank, state) == []
          and seed.is_file()
          and state["superseded_lineages"] == [])

    # identity drift: fail closed, keep file, retry later
    drifted = candidate("drift")
    drifted["sha256"] = "0" * 64
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [drifted]}
    check("reclaim: refuses identity drift (fail closed)",
          reclaim_superseded_lineages(bank, state) == []
          and Path(drifted["path"]).is_file()
          and state["superseded_lineages"] == [drifted])

    # grace window: freshly superseded lineage is kept (in-flight merges)
    fresh = candidate("fresh")
    fresh["superseded_unix"] = time.time()
    state = {"lane": 0, "lineage_path": "current",
             "lineage_sha256": "x", "superseded_lineages": [fresh]}
    check("reclaim: grace window keeps freshly superseded lineage",
          reclaim_superseded_lineages(bank, state) == []
          and Path(fresh["path"]).is_file()
          and state["superseded_lineages"] == [fresh])

    # low-disk fail-closed guard predicate (the run.sh 100GiB precedent)
    check("disk guard: predicate fires only below the bound",
          disk_low(scratch, min_free_bytes=(1 << 60))
          and not disk_low(scratch, min_free_bytes=1)
          and disk_free_bytes(scratch) > 0)


def _synthetic_episode(root: Path, cycle: int, task_id: str, template: str,
                        *, passed: bool, split: str = "train",
                        turn_text: str = "Action: finish\nArguments: {}\nFinal: ok") -> None:
    """Publish minimal honest attempt evidence for the PG batch builder."""
    import hashlib
    import tiktoken

    from ndm.e97_onpolicy_records import canonical_json

    enc = tiktoken.get_encoding("p50k_base")
    cycle_dir = root / "episodes" / f"cycle-{cycle:04d}" / task_id / "attempt"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    text = ("System prompt here\n\nUser: do the task\n\nAssistant:\n"
            + turn_text)
    record = {
        "schema": "emender-rl-loop-episode-v1", "status": "finished",
        "native_record": text,
        "generations": [{"turn": 0, "reason": "valid",
                         "token_ids": enc.encode_ordinary(turn_text)}],
    }
    (cycle_dir / "episode-private.json").write_text(canonical_json(record))
    grade = {"schema": "emender-rl-loop-grade-v1", "task_id": task_id,
             "task_sha256": "0" * 64, "stage": "policy",
             "passed": bool(passed), "checks": []}
    (cycle_dir / "grade.json").write_text(canonical_json(grade))
    summary_path = root / "episodes" / f"cycle-{cycle:04d}" / "collect-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "emender-rl-loop-collect-summary-v1", "cycle": cycle,
               "policy_checkpoint_sha256": "0" * 64, "outcomes": []}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    summary["outcomes"].append({
        "task_id": task_id, "template": template, "split": split,
        "receipt_eligible": split == "train",
        "attempt": {"status": "finished", "grade_passed": bool(passed)},
        "correction": None, "receipt": None, "targets": 0})
    summary_path.write_text(json.dumps(summary, sort_keys=True))


def test_pg_channel(scratch: Path) -> None:
    """v2 policy-gradient channel CPU wiring (builder, selection, receipts,
    lane flag defaults)."""
    from rl_pg_batch import bind_batch_sha, collect_batch, verify_batch
    from rl_pg_screen import apply_selection
    from rl_policy_gradient_step import (PG_RECEIPT_SCHEMA, append_pg_receipt,
                                          pg_stream_tail, verify_pg_stream)

    root = scratch / "pg-episodes"
    # two instances of one sealed family (pass + fail = contrast), one
    # dev-split attempt (must be excluded), one singleton-task group
    _synthetic_episode(root, 1, "lakeexp-aaaa1111", "firstparty-train-direct-read-v1",
                       passed=True)
    _synthetic_episode(root, 1, "lakeexp-bbbb2222", "firstparty-train-direct-read-v1",
                       passed=False)
    _synthetic_episode(root, 1, "lakeexp-cccc3333", "firstparty-development-direct-read-v1",
                       passed=True, split="development")
    _synthetic_episode(root, 1, "lakeexp-dddd4444", "firstparty-train-opaque-read-v1",
                       passed=False)
    body = collect_batch(root, [1], group_by="family")
    manifest = bind_batch_sha(body)
    verify_batch(manifest, reencode=True)
    check("pg-batch family grouping + split isolation",
          {e["group_id"] for e in manifest["entries"]} ==
          {"firstparty-train-direct-read-v1", "firstparty-train-opaque-read-v1"}
          and all(e["split"] == "train" for e in manifest["entries"])
          and manifest["contrast"]
          and manifest["contrastive_groups"] == ["firstparty-train-direct-read-v1"],
          json.dumps(manifest["groups"], sort_keys=True))
    task_body = collect_batch(root, [1], group_by="task")
    check("pg-batch task grouping yields singletons (no false contrast)",
          not task_body["contrast"]
          and all(g["episodes"] == 1 for g in task_body["groups"].values()))
    capped = collect_batch(root, [1], group_by="family", max_episodes=1)
    check("pg-batch recency cap", len(capped["entries"]) == 1)

    # selection rule: S2 ceiling + S4 largest-lr + S5 guard rounding
    plan = {"kl_ceiling": 0.05, "guard_multiplier": 3.0}
    results = [
        {"run_tag": "a", "lr": 1e-6, "kl_beta": 0.01, "completed": True,
         "steps": [{"surrogate_part": 1.0, "realized_kl_vs_old_mean": 0.001},
                   {"surrogate_part": 1.0, "realized_kl_vs_old_mean": 0.001}]},
        {"run_tag": "b", "lr": 2e-6, "kl_beta": 0.01, "completed": True,
         "steps": [{"surrogate_part": 1.0, "realized_kl_vs_old_mean": 0.01},
                   {"surrogate_part": 0.5, "realized_kl_vs_old_mean": 0.01}]},
        {"run_tag": "c", "lr": 1e-5, "kl_beta": 0.01, "completed": True,
         "steps": [{"surrogate_part": 1.0, "realized_kl_vs_old_mean": 0.5},
                   {"surrogate_part": 0.4, "realized_kl_vs_old_mean": 0.5}]},
    ]
    selection = apply_selection(plan, results)
    check("pg-selection S2/S3/S4/S5 (largest sane lr, 3x guard)",
          selection["selected"]["run_tag"] == "b"
          and selection["selected"]["lr"] == 2e-6
          and abs(selection["selected"]["production_kl_guard"] - 0.03) < 1e-9
          and selection["reason"] == "S4"
          and "a" in selection["rule_miss"]["s3_vacuous_run_tags"],
          json.dumps(selection, sort_keys=True))

    # digest-linked PG train-event receipt chain
    stream = scratch / "pg-stream.jsonl"
    d1 = append_pg_receipt(stream, {"schema": PG_RECEIPT_SCHEMA,
                                    "prev_receipt_sha256": None,
                                    "body": {"run_tag": "t1"}})
    d2 = append_pg_receipt(stream, {"schema": PG_RECEIPT_SCHEMA,
                                    "prev_receipt_sha256": d1,
                                    "body": {"run_tag": "t2"}})
    receipts = verify_pg_stream(stream)
    check("pg receipt chain digest-linked",
          pg_stream_tail(stream) == d2 and len(receipts) == 2
          and receipts[1]["receipt"]["body"]["run_tag"] == "t2")
    broken = scratch / "pg-broken.jsonl"
    broken.write_text(json.dumps({"schema": PG_RECEIPT_SCHEMA,
                                  "prev_receipt_sha256": None,
                                  "body": {"run_tag": "x"},
                                  "receipt_sha256": "0" * 64}) + "\n")
    try:
        verify_pg_stream(broken)
        check("pg receipt chain fails closed on digest drift", False)
    except ValueError:
        check("pg receipt chain fails closed on digest drift", True)

    # serialization-form regression: a LEGACY receipt (per_episode emitted
    # with INT keys, >=10 episodes so numeric vs lexicographic key order
    # diverge — the live bank's first hours hit this) must verify via the
    # int-key-numeric form, and the current string-key form must round-trip
    from ndm.e97_onpolicy_records import canonical_json

    legacy_stream = scratch / "pg-legacy.jsonl"
    per_episode = {index: float(index) for index in range(12)}
    legacy_body = {"run_tag": "legacy", "advantages": {"per_episode": per_episode}}
    legacy_receipt = {"schema": PG_RECEIPT_SCHEMA,
                      "prev_receipt_sha256": None,
                      "body": legacy_body}
    digest = hashlib.sha256(
        canonical_json({k: v for k, v in legacy_receipt.items()
                       if k != "receipt_sha256"}).encode("utf-8")).hexdigest()
    legacy_receipt["receipt_sha256"] = digest
    legacy_stream.write_text(canonical_json(legacy_receipt) + "\n")
    verified = verify_pg_stream(legacy_stream)
    check("pg receipt legacy int-key-numeric form verifies",
          verified[0]["serialization_form"] == "int-key-numeric"
          and verified[0]["receipt"]["receipt_sha256"] == digest)
    d3 = append_pg_receipt(legacy_stream, {
        "schema": PG_RECEIPT_SCHEMA,
        "prev_receipt_sha256": digest,
        "body": {"run_tag": "current",
                 "advantages": {"per_episode": {str(i): float(i)
                                                for i in range(12)}}}})
    verified2 = verify_pg_stream(legacy_stream)
    check("pg receipt chain mixes legacy + current forms",
          len(verified2) == 2
          and verified2[0]["serialization_form"] == "int-key-numeric"
          and verified2[1]["serialization_form"] == "string-key-lexicographic"
          and verified2[1]["receipt"]["receipt_sha256"] == d3)

    # lane run-parser: default channel UNCHANGED (sft-receipts) and the
    # policy-gradient choice parses (behavioral: a parse that reaches
    # "lane not initialized" proves the CLI accepted the flags)
    lane_script = Path(__file__).resolve().parent / "rl_bank_lane.py"
    for extra, name in (( [], "default"),
                        (["--lane-train-step", "policy-gradient"], "pg")):
        completed = subprocess.run(
            [sys.executable, str(lane_script), "run", "--bank", str(scratch),
             "--lane", "0", "--args-json", "/dev/null", *extra],
            capture_output=True, text=True, timeout=120)
        check(f"lane run-parser {name} flags accepted",
              completed.returncode == 1
              and "lane not initialized" in completed.stderr,
              f"rc={completed.returncode} stderr={completed.stderr[-120:]!r}")


def test_receipt_exposure(scratch: Path) -> None:
    from unittest.mock import patch
    import rl_build_pack as packer
    from rl_bank_lane import _adopted_exposure_rows, _record_adopted_exposure
    from rl_train_step import sampled_pack_records
    from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
    from rl_common import workspace_paths

    paths = workspace_paths(scratch / "exposure")
    receipts = [{"receipt_sha256": str(i) * 64, "kind": "teacher-corrected",
                 "task_id": str(i), "task_sha256": "a" * 64, "cycle": 1,
                 "policy_checkpoint": {"checkpoint_sha256": "b" * 64},
                 "episode": {"episode_text": "fixture", "generations": [],
                             "supervise_from": 0, "tokens": 1500, "targets": 1499}}
                for i in range(1, 4)]
    consumed = paths["root"] / "consumed-receipts.jsonl"
    argv = ["rl_build_pack.py", "--workspace", str(paths["root"]), "--cycle", "1",
            "--window", "3", "--consumed-ledger", str(consumed),
            "--python", sys.executable]
    with patch.object(sys, "argv", argv), patch.object(packer, "walk_stream", return_value=receipts), \
         patch("scripts.build_e97_pi_native_curriculum.encode_candidate",
               return_value=([1] * 1500, [0] + [1] * 1499, [])):
        packer.main()
    check("E1 packing does not consume any receipts", not consumed.exists())
    packed = [json.loads(line) for line in
              (paths["root"] / "packed-receipts.jsonl").read_text().splitlines()]
    check("E1 packed inventory is separate from exposure", len(packed) == 3)
    authority = paths["packs"] / "cycle-0001" / "authority"
    packs = paths["packs"] / "cycle-0001" / "packs"
    identity = SFTSamplerIdentity(authority_manifest_sha256=_sha(authority / "manifest.json"),
                                  pack_manifest_sha256=_sha(packs / "manifest.json"),
                                  sampler_key=970001, data_world_size=1, context_size=2048)
    data = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0,
                                verify_payload_hashes=True, sampler_mode="epoch-permutation")
    samples = []
    for _ in range(len(data.packs)):
        sample = sampled_pack_records(data, data.next_absolute_rank_sample_index)
        batch = data.get_boundary_aware_batch(1, device="cpu")
        check("E1 sample log names the actual materialized pack " + sample["pack_id"],
              int(batch[-1][0]) == 1499 and len(sample["record_ids"]) == 1)
        samples.append(sample)
    row = {"authority_manifest_sha256": identity.authority_manifest_sha256,
           "pack_manifest_sha256": identity.pack_manifest_sha256,
           "checkpoint_sha256": "c" * 64, "sampled_packs": samples[:1]}
    rows = _adopted_exposure_rows(paths, 1, row)
    check("E1 preparing exposure does not consume a rejected candidate", not consumed.exists())
    _record_adopted_exposure(paths, 1, row)
    exposed = packer.load_consumed_ledger(consumed)
    check("E1 only sampled adopted receipts consumed", len(rows) == len(exposed) == 1)
    remaining = packer.select_window_receipts(receipts, exposed, 32)
    check("E1 unsampled receipts remain eligible", len(remaining) == 2)
    entry = json.loads(consumed.read_text())
    check("E1 consumed entries name sampled pack and adopted checkpoint",
          entry["sampled_pack_ids"] == [samples[0]["pack_id"]]
          and entry["adopted_checkpoint_sha256"] == row["checkpoint_sha256"])
    row["sampled_packs"] = samples
    full_rows = _adopted_exposure_rows(paths, 1, row)
    check("E1 one permutation pass covers every pack and receipt",
          len({r["pack_id"] for r in samples}) == len(data.packs) == 3
          and {r["receipt_sha256"] for r in full_rows} == {r["receipt_sha256"] for r in receipts})
    data.close()


def test_lane_channel_chain(scratch: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    from rl_common import workspace_paths

    paths = workspace_paths(scratch / "channel-chain")
    paths["root"].mkdir(parents=True)
    args = SimpleNamespace(lane=0, args_json=Path("/dev/null"), probe_timeout=10,
                           anchor_period=1, anchor_authority_root=Path("fixture"))
    summary = {"policy_checkpoint_sha256": "parent", "attempts": 1, "outcomes": []}
    state = {"lineage_path": "parent.pt", "lineage_sha256": "parent",
             "updates_total": 4, "anchor_updates_total": 2,
             "train_step_channel": "sft-receipts"}
    order = []
    rows = {}

    def train(channel, _paths, current, *_args):
        order.append(("train", channel, current["lineage_sha256"]))
        row = {"checkpoint": channel + ".pt", "checkpoint_sha256": channel + "-sha",
               "parent_checkpoint_sha256": current["lineage_sha256"], "loss": 1.0}
        rows[channel] = row
        return row

    def spawn(cmd, **kwargs):
        sha = cmd[cmd.index("--checkpoint-sha256") + 1]
        order.append(("probe", sha))
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"frame_valid": True, "checkpoint_sha256": sha}))

    def persist(_root, current):
        order.append(("adopt", current["lineage_sha256"]))

    with patch.object(lane, "_anchor_train_row", side_effect=lambda *a: train("anchor", *a)), \
         patch.object(lane, "_sft_train_row", side_effect=lambda *a: train("receipts", *a)), \
         patch.object(lane, "_spawn", side_effect=spawn), \
         patch.object(lane, "write_lane_state", side_effect=persist), \
         patch.object(lane, "_record_adopted_exposure") as expose, \
         patch("rl_receipts.walk_stream", return_value=[]):
        lane._train_cycle_channels({}, paths, state, args, 1, {}, paths["root"] / "log",
                                  summary, "0", 1)
    check("E2 receipts parent is the adopted anchor child sha",
          rows["receipts"]["parent_checkpoint_sha256"] == "anchor-sha")
    check("E2 train probe adopt completes before next channel",
          order == [("train", "anchor", "parent"), ("probe", "anchor-sha"),
                    ("adopt", "anchor-sha"), ("train", "receipts", "anchor-sha"),
                    ("probe", "receipts-sha"), ("adopt", "receipts-sha")])
    check("E2 retained chain counts both adopted updates",
          state["lineage_sha256"] == "receipts-sha" and state["updates_total"] == 6
          and state["anchor_updates_total"] == 3
          and [r["sha256"] for r in state["superseded_lineages"]] == ["parent", "anchor-sha"])
    metrics = json.loads((paths["cycles"] / "cycle-0001" / "metrics.json").read_text())
    check("E2 metrics trains array preserves both channels",
          [r["train_step_channel"] for r in metrics["trains"]] ==
          ["sft-anchor-corpus", "sft-receipts"])
    check("E2 only adopted receipts channel commits exposure", expose.call_count == 1)


def test_invalid_probe_rejection(scratch: Path) -> None:
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    from rl_common import workspace_paths

    args = SimpleNamespace(lane=0, args_json=Path("/dev/null"), probe_timeout=10)
    summary = {"policy_checkpoint_sha256": "parent", "attempts": 1, "outcomes": []}
    original_spawn = lane._spawn
    for index, (channel, frame_valid) in enumerate([
            ("sft-anchor-corpus", False), ("sft-receipts", False),
            ("policy-gradient", False), ("sft-receipts", "true"),
            ("sft-receipts", None)]):
        paths = workspace_paths(scratch / f"probe-reject-{index}")
        paths["root"].mkdir(parents=True)
        checkpoint = paths["root"] / "rejected.pt"
        checkpoint.write_text("unadopted checkpoint")
        state = {"lineage_path": "parent.pt", "lineage_sha256": "parent",
                 "updates_total": 4, "anchor_updates_total": 2,
                 "pg_updates_total": 1, "superseded_lineages": []}
        before = deepcopy(state)
        row = {"checkpoint": str(checkpoint), "checkpoint_sha256": "candidate", "loss": 1.0}

        def zero_exit_probe(cmd, **kwargs):
            out = cmd[cmd.index("--output") + 1]
            record = {"checkpoint_sha256": "candidate"}
            if frame_valid is not None:
                record["frame_valid"] = frame_valid
            script = ("from pathlib import Path; import json; "
                      f"p=Path({out!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                      f"p.write_text(json.dumps({record!r}))")
            original_spawn([sys.executable, "-c", script], **kwargs)

        with patch.object(lane, "_spawn", side_effect=zero_exit_probe), \
             patch.object(lane, "write_lane_state") as persist, \
             patch.object(lane, "_record_adopted_exposure") as expose, \
             patch("rl_receipts.walk_stream", return_value=[]):
            adopted = lane._probe_adopt_train_row(
                {}, paths, state, args, 1, os.environ, paths["root"] / "log",
                summary, "0", row, channel, row_idx=0)
        check(f"E3 zero-exit invalid probe rejects {channel} ({frame_valid!r})",
              adopted is False and state == before and not persist.called and not expose.called)
        metrics = json.loads((paths["cycles"] / "cycle-0001" / "metrics.json").read_text())
        check(f"E3 rejected candidate logged without exposure {index}",
              metrics["trains"][0]["adopted"] is False
              and not (paths["root"] / "consumed-receipts.jsonl").exists()
              and not checkpoint.exists())


def test_fresh_solve_first_action(scratch: Path) -> None:
    from rl_bank_lane import _correction_requires_fresh_solve
    from rl_loop_driver import degeneracy_screen, retained_prefix
    from scripts.e97_first_party_validator_first_action import _check_first_action
    from scripts.e97_pi_native_codec import semantic_turn

    spec_path = scratch / "first-action-spec.json"
    required = {"tool": "read", "arguments": {"path": "correct.txt"}}
    spec_path.write_text(json.dumps({"required_first_action": required}))
    body = {"task_lake": {"validator": {"spec_path": str(spec_path),
                                        "spec_sha256": _sha(spec_path)}}}
    tools = [{"name": name, "label": name, "description": "fixture",
              "parameters": {"type": "object"}} for name in ("read", "bash")]

    def action(name, arguments, analysis="clean distinct analysis"):
        return {"role": "assistant", "reasoning_content": analysis,
                "tool_calls": [{"type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments)}}]}

    def record(first):
        return {"source_messages": [first, {"role": "toolResult", "isError": False,
                  "content": [{"type": "text", "text": "clean observation"}]},
                  action("finish", {"message": "wrong final"}, "different final analysis")]}

    for name, arguments, label in [("bash", {"command": "cat correct.txt"}, "tool"),
                                    ("read", {"path": "wrong.txt"}, "argument")]:
        policy = record(action(name, arguments))
        check(f"E5 wrong first {label} is clean but requires fresh solve",
              degeneracy_screen(policy)[0]
              and _correction_requires_fresh_solve(body, policy, tools))
        # The bank passes no prefix for fresh solves; run_episode then has no
        # retained prefix metadata, so the correction's supervised span starts at 0.
        fresh = _correction_requires_fresh_solve(body, policy, tools)
        prefix = None if fresh else policy["source_messages"]
        frames = 0 if prefix is None else len(retained_prefix(prefix, tools)[0])
        teacher = action("read", {"path": "correct.txt"})
        corrected_first = semantic_turn(teacher if fresh else prefix[0], tools)
        _check_first_action([{"tool_name": corrected_first["name"],
                              "arguments": corrected_first["arguments"]}], required)
        check(f"E5 fresh {label} correction supervises replaceable first action from zero",
              prefix is None and frames == 0)
    recoverable = record(action("read", {"path": "correct.txt", "offset": 1}))
    check("E5 correct first action with wrong final keeps repair continuation",
          not _correction_requires_fresh_solve(body, recoverable, tools))
    no_action = {"source_messages": [action("finish", {"message": "wrong"})]}
    check("E5 dropped finish leaves first action recoverable",
          not _correction_requires_fresh_solve(body, no_action, tools))
    unanswered = {"source_messages": [action("read", {"path": "wrong.txt"})]}
    check("E5 dropped unresolved action leaves first action recoverable",
          not _correction_requires_fresh_solve(body, unanswered, tools))
    think_first = record(action("think", {"thought": "private"}))
    check("E5 think is not a sealed first action",
          not _correction_requires_fresh_solve(body, think_first, tools))
    degenerate = record(action("read", {"path": "correct.txt"}))
    degenerate["source_messages"][-1]["reasoning_content"] = "clean distinct analysis"
    check("E5 degeneracy still requires fresh solve",
          _correction_requires_fresh_solve(body, degenerate, tools))
    spec_path.write_text("{}")
    body["task_lake"]["validator"]["spec_sha256"] = _sha(spec_path)
    check("E5 absent first-action criterion preserves clean repairs",
          not _correction_requires_fresh_solve(body, recoverable, tools))
    check("E5 seed tasks retain clean repair behavior",
          not _correction_requires_fresh_solve({}, recoverable, tools))


def test_fresh_solve_collect(scratch: Path) -> None:
    """Exercise actual collection routing and receipt supervision, CPU stubs only."""
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane

    tools = [{"name": name, "label": name, "description": "fixture",
              "parameters": {"type": "object"}} for name in ("read", "bash")]
    for index, (name, arguments, fresh) in enumerate([
            ("bash", {"command": "cat correct.txt"}, True),
            ("read", {"path": "wrong.txt"}, True),
            ("read", {"path": "correct.txt"}, False)]):
        root = scratch / f"fresh-collect-{index}"
        bank = bank_paths(root)
        ensure_bank_layout(bank)
        seed = root / "seed.pt"
        _synthetic_checkpoint(seed, 1.0, "seed")
        init_lane_state(bank, 0, checkpoint=seed, checkpoint_sha256=_sha(seed), note="unit")
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps({"required_first_action": {
            "tool": "read", "arguments": {"path": "correct.txt"}}}))
        body = {"template": "fixture", "prompt": "read correct.txt", "receipt_eligible": True,
                "workspace_files": {"correct.txt": "original"},
                "task_lake": {"limits": {"turns": 4}, "validator": {
                    "spec_path": str(spec_path), "spec_sha256": _sha(spec_path)}}}
        task = {"task_id": "fixture", "body": body, "task_sha256": "a" * 64}
        first = {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}
        policy = {"status": "stopped", "native_record": "fixture", "generations": [],
                  "source_messages": [first, {"role": "toolResult", "isError": False,
                    "content": [{"type": "text", "text": "clean"}]}]}
        episodes = []
        receipts = []

        def run_episode(**kwargs):
            episodes.append(kwargs)
            if len(episodes) == 1:
                (kwargs["workspace"] / "correct.txt").write_text("policy mutation")
                (kwargs["episode_dir"] / "episode-private.json").write_text(json.dumps(policy))
                return policy
            prefix = kwargs["prefix_messages"]
            return {"status": "finished", "close_verified": True, "generations": [],
                    "prefix": None if prefix is None else {"retained_frames": 1}}

        def receipt(**kwargs):
            receipts.append(kwargs)
            return {"receipt_sha256": "b" * 64, "episode": {"targets": 64}}

        args = SimpleNamespace(bank=root, lane=0, cycle=1, args_json=Path("/dev/null"),
                               teacher="fixture", teacher_model="fixture", preclaimed=None,
                               max_tasks=1, claim_ttl=60, teacher_min_interval=0)
        replacements = {"load_pilot": SimpleNamespace(Metrics=lambda: SimpleNamespace(calls=[])),
                        "load_curriculum": SimpleNamespace(SYSTEM="fixture"),
                        "tool_manifest": {"pi_bin": "/dev/null", "model_visible_tools": tools},
                        "load_policy_engine": None, "make_policy_generate": None,
                        "make_teacher_generate": None, "claim_pool_task": task,
                        "heartbeat_claim": True, "retire_task": None,
                        "append_receipt": "b" * 64}
        with ExitStack() as stack:
            for attribute, value in replacements.items():
                stack.enter_context(patch.object(lane, attribute, return_value=value))
            stack.enter_context(patch.object(lane, "run_episode", side_effect=run_episode))
            stack.enter_context(patch.object(lane, "grade_episode", side_effect=[(False, {}), (True, {})]))
            stack.enter_context(patch.object(lane, "_receipt_from_episode", side_effect=receipt))
            lane.bank_collect(args)
        correction = episodes[1]
        check(f"E5 collect routes correction prefix and supervision {index}",
              (correction["prefix_messages"] is None) == fresh
              and receipts[0]["supervise_from"] == (0 if fresh else 1)
              and receipts[0]["teacher"]["continuation"] ==
                  ("fresh-episode" if fresh else "same-transcript-splice"))
        check(f"E5 collect resets only irreversibly wrong workspaces {index}",
              (correction["workspace"] != episodes[0]["workspace"]) == fresh
              and (correction["workspace"] / "correct.txt").read_text() ==
                  ("original" if fresh else "policy mutation"))


def test_no_signal_exposure_skip(scratch: Path) -> None:
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    from rl_common import workspace_paths

    paths = workspace_paths(scratch / "no-signal")
    paths["root"].mkdir(parents=True)
    consumed = paths["root"] / "consumed-receipts.jsonl"
    consumed.write_text(json.dumps({"receipt_sha256": "previous"}) + "\n")
    before_ledger = consumed.read_bytes()
    state = {"lineage_path": "parent.pt", "lineage_sha256": "parent", "updates_total": 4}
    before_state = deepcopy(state)
    args = SimpleNamespace(lane=0, args_json=Path("/dev/null"), pack_timeout=10, train_timeout=10)
    candidate = paths["root"] / "low-loss.pt"

    def spawn(cmd, **kwargs):
        if cmd[1].endswith("rl_build_pack.py"):
            cycle_packs = paths["packs"] / "cycle-0001"
            cycle_packs.mkdir(parents=True)
            (cycle_packs / "build-summary.json").write_text(json.dumps({
                "authority_manifest_sha256": "authority", "pack_manifest_sha256": "packs",
                "context_size": 2048}))
        else:
            candidate.write_text("unadopted checkpoint")
            log = Path(cmd[cmd.index("--log-jsonl") + 1])
            log.parent.mkdir(parents=True)
            log.write_text(json.dumps({"event": "checkpoint", "checkpoint": str(candidate),
                                       "checkpoint_sha256": "low-loss", "loss": 0.1}) + "\n")

    with patch.object(lane, "_spawn", side_effect=spawn), \
         patch.object(lane, "_record_adopted_exposure") as expose:
        row = lane._sft_train_row(paths, state, args, 1, {}, paths["root"] / "log")
    check("E1 no-signal skipped training advances no lineage counters or consumption",
          row is None and state == before_state and consumed.read_bytes() == before_ledger
          and not expose.called)
    check("E1 no-signal checkpoint is reclaimed", not candidate.exists())


def test_rejected_anchor_chain(scratch: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    from rl_common import workspace_paths

    paths = workspace_paths(scratch / "rejected-anchor-chain")
    paths["root"].mkdir(parents=True)
    args = SimpleNamespace(lane=0, args_json=Path("/dev/null"), probe_timeout=10,
                           anchor_period=1, anchor_authority_root=Path("fixture"))
    summary = {"policy_checkpoint_sha256": "parent", "attempts": 1, "outcomes": []}
    state = {"lineage_path": "parent.pt", "lineage_sha256": "parent",
             "updates_total": 4, "anchor_updates_total": 2}
    rows = {}

    def train(channel, _paths, current, *_args):
        row = {"checkpoint": channel + ".pt", "checkpoint_sha256": channel + "-sha",
               "parent_checkpoint_sha256": current["lineage_sha256"], "loss": 1.0}
        rows[channel] = row
        return row

    def spawn(cmd, **kwargs):
        sha = cmd[cmd.index("--checkpoint-sha256") + 1]
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"frame_valid": sha != "anchor-sha", "checkpoint_sha256": sha}))

    with patch.object(lane, "_anchor_train_row", side_effect=lambda *a: train("anchor", *a)), \
         patch.object(lane, "_sft_train_row", side_effect=lambda *a: train("receipts", *a)), \
         patch.object(lane, "_spawn", side_effect=spawn), \
         patch.object(lane, "write_lane_state"), \
         patch.object(lane, "_unlink_orphan_checkpoint"), \
         patch.object(lane, "_record_adopted_exposure") as expose, \
         patch("rl_receipts.walk_stream", return_value=[]):
        lane._train_cycle_channels({}, paths, state, args, 1, {}, paths["root"] / "log",
                                  summary, "0", 1)
    check("E2 E3 receipts parent excludes rejected anchor child",
          rows["receipts"]["parent_checkpoint_sha256"] == "parent"
          and state["updates_total"] == 5 and state["anchor_updates_total"] == 2)
    metrics = json.loads((paths["cycles"] / "cycle-0001" / "metrics.json").read_text())
    check("E3 rejected anchor and adopted receipts both remain in metrics",
          [r["adopted"] for r in metrics["trains"]] == [False, True] and expose.call_count == 1)


def test_channel_checkpoint_paths(scratch: Path) -> None:
    """Equal rounded-loss filenames must not overwrite an adopted anchor."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    from rl_common import workspace_paths

    paths = workspace_paths(scratch / "checkpoint-paths")
    paths["root"].mkdir(parents=True)
    state = {"lineage_path": "parent.pt", "lineage_sha256": "parent"}
    args = SimpleNamespace(lane=0, args_json=Path("/dev/null"), pack_timeout=10,
                           train_timeout=10, anchor_keys="1", anchor_authority_root=Path("fixture"),
                           anchor_authority_sha256="anchor-authority", anchor_pack_root=Path("fixture"),
                           anchor_pack_sha256="anchor-packs")
    commands = []

    def spawn(cmd, **kwargs):
        if cmd[1].endswith("rl_build_pack.py"):
            cycle_packs = paths["packs"] / "cycle-0001"
            cycle_packs.mkdir(parents=True)
            (cycle_packs / "build-summary.json").write_text(json.dumps({
                "authority_manifest_sha256": "receipts-authority",
                "pack_manifest_sha256": "receipts-packs", "context_size": 2048}))
            return
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output-root") + 1])
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / "checkpoint_agent_sft_u000001_loss_1.0000.pt"
        checkpoint.write_text("anchor" if len(commands) == 1 else "receipts")
        log = Path(cmd[cmd.index("--log-jsonl") + 1])
        with log.open("a") as stream:
            stream.write(json.dumps({"event": "checkpoint", "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": _sha(checkpoint), "loss": 1.0}) + "\n")

    with patch.object(lane, "_spawn", side_effect=spawn), \
         patch.object(lane, "_adopted_exposure_rows", return_value=[]):
        anchor = lane._anchor_train_row(paths, state, args, 1, {}, paths["root"] / "log")
        state.update(lineage_path=anchor["checkpoint"], lineage_sha256=anchor["checkpoint_sha256"])
        receipts = lane._sft_train_row(paths, state, args, 1, {}, paths["root"] / "log")
    anchor_path, receipts_path = Path(anchor["checkpoint"]), Path(receipts["checkpoint"])
    cycle_training = paths["training"] / "cycle-0001"
    check("E2 anchor and receipts commands use disjoint checkpoint output roots",
          anchor_path.parent == cycle_training / "anchor-checkpoints"
          and receipts_path.parent == cycle_training / "checkpoints"
          and commands[0][commands[0].index("--output-root") + 1] !=
              commands[1][commands[1].index("--output-root") + 1])
    check("E2 equal rounded-loss basenames retain both checkpoint identities",
          anchor_path.name == receipts_path.name and anchor_path != receipts_path
          and anchor_path.read_text() == "anchor" and _sha(anchor_path) == anchor["checkpoint_sha256"]
          and receipts_path.read_text() == "receipts")
    check("E2 receipts command names preserved anchor checkpoint parent",
          commands[1][commands[1].index("--parent-checkpoint") + 1] == str(anchor_path)
          and commands[1][commands[1].index("--parent-sha256") + 1] == _sha(anchor_path))


def test_block_training(scratch: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    import rl_build_pack as packer
    from rl_common import workspace_paths
    from rl_train_step import full_pass_sample_counts, train_pack_update
    from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity

    args = SimpleNamespace(lane=0, args_json=Path('/dev/null'), probe_timeout=10,
                           pack_timeout=60, train_timeout=60, anchor_period=1,
                           anchor_authority_root=Path('fixture'), block_mode='on',
                           block_min_targets=65536, block_max_targets=131072,
                           block_max_wait_cycles=2)
    inventory = [{'receipt_sha256': f'{i:064x}', 'kind': 'teacher-corrected' if i % 2 else
                  'on-policy-success', 'task_id': str(i), 'task_sha256': 'a' * 64, 'cycle': 1,
                  'policy_checkpoint': {'checkpoint_sha256': 'b' * 64},
                  'episode': {'episode_text': 'fixture', 'generations': [],
                              'supervise_from': 0, 'tokens': 1500, 'targets': 1499}}
                 for i in range(1, 37)]
    captured = []
    run_argv = ['rl_bank_lane.py', 'run', '--bank', str(scratch), '--lane', '0',
                '--args-json', '/dev/null']
    with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', run_argv), \
         patch.object(lane, 'lane_run', side_effect=lambda parsed: captured.append(parsed)):
        lane.main()
    check('F3 BLOCK_MODE defaults off with registered limits',
          (captured[-1].block_mode, captured[-1].block_min_targets,
           captured[-1].block_max_targets, captured[-1].block_max_wait_cycles) ==
          ('off', 65536, 131072, 48))
    with patch.dict(os.environ, {'BLOCK_MODE': 'on', 'BLOCK_MIN_TARGETS': '123',
                               'BLOCK_MAX_TARGETS': '456', 'BLOCK_MAX_WAIT_CYCLES': '7'}, clear=True), \
         patch.object(sys, 'argv', run_argv), \
         patch.object(lane, 'lane_run', side_effect=lambda parsed: captured.append(parsed)):
        lane.main()
    check('F3 all four environment controls bind to run configuration',
          (captured[-1].block_mode, captured[-1].block_min_targets,
           captured[-1].block_max_targets, captured[-1].block_max_wait_cycles) ==
          ('on', 123, 456, 7))
    trigger_state = {}
    lane._initialize_block_state(trigger_state, args)
    threshold_inventory = [{'episode': {'targets': 65536}}]
    check('F3 block trigger fires exactly at threshold',
          lane._block_trigger(trigger_state, args, 1, threshold_inventory) == 'targets')
    trigger_state = {}
    lane._initialize_block_state(trigger_state, args)
    check('F3 below threshold accumulates without early firing',
          lane._block_trigger(trigger_state, args, 1, [{'episode': {'targets': 65535}}]) is None
          and trigger_state['block_wait_cycles'] == 1)
    check('F3 inventory reread in one cycle does not double count wait',
          lane._block_trigger(trigger_state, args, 1, [{'episode': {'targets': 65535}}]) is None
          and trigger_state['block_wait_cycles'] == 1)
    check('F3 empty inventory never fires max wait',
          all(lane._block_trigger(trigger_state, args, c, []) is None for c in range(1, 5)))
    check('F3 teacher priority and strict cap retain remainder',
          [r['task_id'] for r in packer.select_block_receipts(inventory, set(), 2998)] == ['1', '3'])
    check('F3 consumed inventory excluded from full block',
          len(packer.select_block_receipts(inventory, {inventory[0]['receipt_sha256']}, 131072)) == 35)
    check('F3 density formula bounded and predeclared',
          [lane.block_steps(n) for n in (1, 11200, 11201, 65536, 131072, 1000000)] ==
          [8, 8, 9, 47, 94, 256])
    check('F3 full pass budget covers more packs than updates without overrun',
          full_pass_sample_counts(100, 8) == [13, 13, 13, 13, 12, 12, 12, 12]
          and full_pass_sample_counts(3, 8) == [1] * 8)

    paths = workspace_paths(scratch / 'block-training')
    paths['root'].mkdir(parents=True)
    state = {'schema': 'emender-rl-loop-bank-lane-state-v1',
             'lineage_path': 'parent.pt', 'lineage_sha256': 'parent', 'updates_total': 0}
    summary = {'policy_checkpoint_sha256': 'parent', 'attempts': 1, 'outcomes': []}
    commands = []
    block_parents = []
    block_coverages = []
    optimizer_steps = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, tokens, *, loss_mask, **kwargs):
            return self.weight.square() * loss_mask.sum()

    def spawn(cmd, **kwargs):
        commands.append(cmd)
        if cmd[1].endswith('rl_build_pack.py'):
            argv = [cmd[1], *cmd[2:], '--python', sys.executable]
            with patch.object(sys, 'argv', argv), \
                 patch.object(packer, 'walk_stream', return_value=inventory), \
                 patch('scripts.build_e97_pi_native_curriculum.encode_candidate',
                       return_value=([1] * 1500, [0] + [1] * 1499, [])):
                packer.main()
        elif cmd[1].endswith('rl_train_step.py'):
            def option(name):
                return cmd[cmd.index(name) + 1]
            block_parents.append(option('--parent-sha256'))
            authority, packs = Path(option('--authority-root')), Path(option('--pack-root'))
            identity = SFTSamplerIdentity(authority_manifest_sha256=option('--authority-sha256'),
                                          pack_manifest_sha256=option('--pack-sha256'),
                                          sampler_key=970001, data_world_size=1, context_size=2048)
            data = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0,
                                         sampler_mode='epoch-permutation')
            steps = int(option('--steps'))
            model = TinyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            samples = []
            for count in full_pass_sample_counts(len(data.packs), steps):
                result = train_pack_update(model, optimizer, data, sample_count=count,
                                           device=torch.device('cpu'), grad_clip=1.0)
                samples.extend(result['sampled_packs'])
            optimizer_steps.append((steps, float(model.weight.detach()), bool(optimizer.state)))
            block_coverages.append(len({r for s in samples for r in s['record_ids']}))
            checkpoint = Path(option('--output-root')) / 'block.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text('trained block')
            event = {'event': 'checkpoint', 'checkpoint': str(checkpoint),
                     'checkpoint_sha256': _sha(checkpoint), 'loss': 1.0,
                     'authority_manifest_sha256': identity.authority_manifest_sha256,
                     'pack_manifest_sha256': identity.pack_manifest_sha256,
                     'sampled_packs': samples}
            Path(option('--log-jsonl')).write_text(json.dumps(event) + '\n')
            data.close()
        elif 'probe' in cmd:
            out = Path(cmd[cmd.index('--output') + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            candidate = cmd[cmd.index('--checkpoint-sha256') + 1]
            out.write_text(json.dumps({'frame_valid': state['lane_cycle'] != 2 or
                                      candidate.startswith('anchor'),
                                      'checkpoint_sha256': candidate}))

    def anchor(_paths, current, _args, cycle, *_rest):
        return {'checkpoint': f'anchor-{cycle}.pt', 'checkpoint_sha256': f'anchor-{cycle}',
                'loss': 1.0}

    with patch.object(lane, '_spawn', side_effect=spawn), \
         patch.object(lane, '_anchor_train_row', side_effect=anchor), \
         patch.object(packer, 'walk_stream', return_value=inventory):
        for cycle in range(1, 5):
            state['lane_cycle'] = cycle
            lane._train_cycle_channels({}, paths, state, args, cycle, {}, paths['root'] / 'log',
                                      summary, '0', 0 if cycle > 1 else 36)
            if cycle == 1:
                check('F3 anchors continue and no receipts train during accumulation',
                      state['lineage_sha256'] == 'anchor-1' and state['updates_total'] == 1
                      and not commands[:-1] and state['block_inventory_targets'] == 53964)
            if cycle == 2:
                check('F3 rejected block preserves inventory and consumes nothing',
                      state['lineage_sha256'] == 'anchor-2' and state['updates_total'] == 2
                      and state['block_skipped_total'] == state['block_backoff_count'] == 1
                      and state['block_inventory_targets'] == 53964
                      and not (paths['root'] / 'consumed-receipts.jsonl').exists())
            if cycle == 3:
                check('F3 rejection cooldown retains anchors without block retry',
                      state['lineage_sha256'] == 'anchor-3' and state['updates_total'] == 3
                      and len(block_parents) == 1)
    check('F3 blocks pack entire inventory beyond old 32-receipt window',
          block_coverages == [36, 36] and all('--all-unconsumed' in c for c in commands
                                              if c[1].endswith('rl_build_pack.py')))
    check('F3 retry uses current adopted anchor and advances block lineage once',
          block_parents == ['anchor-2', 'anchor-4'] and state['updates_total'] == 5
          and state['anchor_updates_total'] == 4 and state['block_adopted_total'] == 1
          and state['block_attempts_total'] == 2 and state['block_inventory_targets'] == 0
          and state['block_backoff_count'] == 0)
    check('F3 adopted block consumption names every sampled receipt exactly once',
          packer.load_consumed_ledger(paths['root'] / 'consumed-receipts.jsonl') ==
          {r['receipt_sha256'] for r in inventory}
          and len((paths['root'] / 'consumed-receipts.jsonl').read_text().splitlines()) == 36)
    check('F3 one invocation per block retains optimizer internally at density budget',
          all('--full-pass' in c for c in commands if c[1].endswith('rl_train_step.py'))
          and len(optimizer_steps) == 2
          and all(steps == 39 and weight < 1 and persisted
                  for steps, weight, persisted in optimizer_steps))
    # Exercise serial accumulation on the same real CPU authority with packs > steps.
    authority = paths['packs'] / 'cycle-0004' / 'authority'
    packs = paths['packs'] / 'cycle-0004' / 'packs'
    identity = SFTSamplerIdentity(authority_manifest_sha256=_sha(authority / 'manifest.json'),
                                  pack_manifest_sha256=_sha(packs / 'manifest.json'), sampler_key=970001,
                                  data_world_size=1, context_size=2048)
    data = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0,
                                 sampler_mode='epoch-permutation')
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_pack_update(model, optimizer, data, sample_count=5,
                               device=torch.device('cpu'), grad_clip=0)
    check('F3 serial accumulation normalizes by all targets and logs every pack',
          result['targets'] == 5 * 1499 and len(result['sampled_packs']) == 5
          and abs(float(model.weight.detach()) - 0.98) < 1e-6 and result['loss'] == 1.0)
    data.close()
    skip_paths = workspace_paths(scratch / 'block-no-signal')
    skip_paths['root'].mkdir(parents=True)
    skip_state = {'schema': 'emender-rl-loop-bank-lane-state-v1',
                  'lineage_path': 'parent', 'updates_total': 0}
    thin = [{'episode': {'targets': 65536}}]
    with patch.object(lane, '_block_inventory', return_value=thin), \
         patch.object(lane, '_sft_train_row', return_value=None), \
         patch.object(lane, '_record_adopted_exposure') as expose:
        lane._run_receipts_block(skip_paths, skip_state, args, 1, {}, skip_paths['root'] / 'log',
                                 lambda row: check('F3 skipped block must not be probed', False))
    check('F3 low-signal block skips without consumption or lineage advance',
          skip_state['block_inventory_targets'] == 65536 and skip_state['updates_total'] == 0
          and skip_state['block_backoff_count'] == 1 and skip_state['block_skipped_total'] == 1
          and not expose.called and not (skip_paths['root'] / 'consumed-receipts.jsonl').exists())


def test_parallel_collection(scratch: Path) -> None:
    """F4: parallel collection feeds the single learner's block channel.

    Collector lanes append verified receipts to their own lane-local streams;
    with --streams-root (BLOCK_STREAMS_ROOT / the bank lanes default) the
    learner's selection pool becomes the walk_stream-verified UNION of every
    lanes/lane-0N stream, deduped by receipt sha, FIFO by created_unix with
    deterministic (lane, stream-position) tiebreaks, teacher-corrected first.
    The consumed ledger stays at the learner lane; its lines name the source
    lane ONLY on cross-stream lines (absent flag = byte-identical behavior).
    COLLECTOR_ONLY lanes form and chain receipts but never train, adopt a
    merge, or advance their init-seed lineage.
    """
    from contextlib import redirect_stdout
    from io import StringIO
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import hashlib

    import tiktoken

    import rl_bank_lane as lane
    import rl_build_pack as packer
    from ndm.e97_onpolicy_records import canonical_json
    from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
    from rl_common import workspace_paths
    from rl_receipts import append_receipt, build_receipt, walk_stream
    from rl_train_step import sampled_pack_records
    from scripts.build_e97_pi_native_curriculum import encode_candidate

    enc = tiktoken.get_encoding("p50k_base")

    # ---- parser: the new controls are env-bindable with defaults off
    run_argv = ["rl_bank_lane.py", "run", "--bank", str(scratch), "--lane", "0",
                "--args-json", "/dev/null"]
    captured = []
    with patch.dict(os.environ, {"COLLECTOR_ONLY": "1",
                                 "BLOCK_STREAMS_ROOT": "/tmp/streams"}, clear=True), \
         patch.object(sys, "argv", run_argv), \
         patch.object(lane, "lane_run", side_effect=lambda parsed: captured.append(parsed)):
        lane.main()
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(sys, "argv", run_argv), \
         patch.object(lane, "lane_run", side_effect=lambda parsed: captured.append(parsed)):
        lane.main()
    check("F4 collector-only and block-streams-root env controls bind with defaults off",
          captured[0].collector_only is True
          and captured[0].block_streams_root == Path("/tmp/streams")
          and captured[-1].collector_only is False
          and captured[-1].block_streams_root is None)
    check("F4 block streams root defaults to the bank lanes dir only in block mode",
          lane._block_streams_root(SimpleNamespace(block_mode="on", block_streams_root=None,
                                                   bank="/tmp/bank")) == Path("/tmp/bank/lanes")
          and lane._block_streams_root(SimpleNamespace(
              block_mode="on", block_streams_root=Path("/x"), bank="/tmp/bank")) == Path("/x")
          and lane._block_streams_root(SimpleNamespace(
              block_mode="off", block_streams_root=Path("/x"), bank="/tmp/bank")) is None)

    # ---- a mini single-learner bank: lane-00 learner, lanes 01-03 collectors
    episode_text = "User:\nhi\n\nAssistant:\nhello\n"
    turn_ids = enc.encode("hello")
    ids, episode_mask, _ = encode_candidate(episode_text, [{"token_ids": turn_ids}], 0, enc)

    def mint(prev, *, kind, created, task_id):
        receipt = build_receipt(
            cycle=1, task={"task_id": task_id, "task_sha256": "a" * 64}, kind=kind,
            policy_checkpoint={"checkpoint_sha256": "seed" + "0" * 60},
            episode_text=episode_text, generations=[{"token_ids": turn_ids}],
            supervise_from=0, tokens=len(ids), targets=int(sum(episode_mask)),
            grade={"passed": True, "score": 1.0},
            teacher=({"model": "fixture-teacher", "checkpoint_sha256": "d" * 64}
                     if kind == "teacher-corrected" else None),
            attempt_link=({"receipt_sha256": "b" * 64}
                          if kind == "teacher-corrected" else None),
            prev_receipt_sha256=prev)
        receipt["created_unix"] = created
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt["receipt_sha256"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")).hexdigest()
        return receipt

    bank = bank_paths(scratch / "parallel-bank")
    ensure_bank_layout(bank)
    learner = lane_paths(bank, 0)
    collector1 = lane_paths(bank, 1)
    collector2 = lane_paths(bank, 2)
    collector3 = lane_paths(bank, 3)
    a = mint(None, kind="teacher-corrected", created=1000.0, task_id="A")
    append_receipt(learner, a)
    b = mint(a["receipt_sha256"], kind="on-policy-success", created=1001.0, task_id="B")
    append_receipt(learner, b)
    append_receipt(collector1, a)  # byte-identical duplicate across streams
    c = mint(a["receipt_sha256"], kind="teacher-corrected", created=1000.5, task_id="C")
    append_receipt(collector1, c)
    f = mint(None, kind="teacher-corrected", created=1000.5, task_id="F")
    append_receipt(collector2, f)
    f2 = mint(f["receipt_sha256"], kind="teacher-corrected", created=1000.5, task_id="F2")
    append_receipt(collector2, f2)
    g = mint(None, kind="on-policy-success", created=1000.2, task_id="G")
    append_receipt(collector3, g)
    # decoys: a corrupt stream under a non-numeric lane, an empty lane, a file
    decoy = bank["lanes"] / "lane-xx"
    (decoy / "receipts").mkdir(parents=True)
    (decoy / "receipts" / "stream.jsonl").write_text("not a receipt\n")
    (bank["lanes"] / "lane-04").mkdir()
    (bank["lanes"] / "lane-05").write_text("a file, not a lane")

    union = packer.load_selection_receipts(learner, enc=enc, streams_root=bank["lanes"])
    check("F4 cross-stream union verifies every lane stream and skips non-lane decoys",
          len(union) == 6
          and {r["task_id"] for r in union} == {"A", "B", "C", "F", "F2", "G"})
    check("F4 cross-stream FIFO dedupes by receipt sha with lane and position tiebreaks",
          [r["task_id"] for r in union] == ["A", "G", "C", "F", "F2", "B"]
          and len({r["receipt_sha256"] for r in union}) == 6)
    check("F4 source_lane annotations name the originating lane stream",
          {r["task_id"]: r["source_lane"] for r in union} ==
          {"A": "lane-00", "B": "lane-00", "C": "lane-01", "F": "lane-02",
           "F2": "lane-02", "G": "lane-03"})
    check("F4 window selection keeps teacher-corrected priority across streams",
          [r["task_id"] for r in packer.select_window_receipts(union, set(), 6)] ==
          ["A", "C", "F", "F2", "G", "B"])
    own = packer.load_selection_receipts(learner, enc=enc)
    check("F4 unflagged selection is byte-identical single-stream behavior",
          own == walk_stream(learner, enc=enc)
          and all("source_lane" not in r for r in own)
          and packer.load_selection_receipts(collector3, enc=enc) == walk_stream(collector3, enc=enc)
          and packer.load_selection_receipts(
              workspace_paths(bank["lanes"] / "lane-04"), enc=enc) == [])
    try:
        packer.load_selection_receipts(learner, enc=enc, streams_root=scratch / "absent")
        missing_root_failed = False
    except SystemExit:
        missing_root_failed = True
    check("F4 streams root fails closed on a missing directory", missing_root_failed)
    inventory = lane._block_inventory(learner, lane._block_streams_root(
        SimpleNamespace(block_mode="on", block_streams_root=None, bank=bank["root"])))
    check("F4 block inventory spans collector streams from the bank lanes root",
          [r["task_id"] for r in inventory] == ["A", "C", "F", "F2", "G", "B"]
          and all("source_lane" in r for r in inventory))
    check("F4 single-stream block inventory keeps lane-local order without annotations",
          [r["task_id"] for r in lane._block_inventory(learner)] == ["A", "B"]
          and all("source_lane" not in r for r in lane._block_inventory(learner)))

    # ---- end to end: the learner packs the union; ledgers stay lane-local
    consumed_ledger = learner["root"] / "consumed-receipts.jsonl"
    argv = ["rl_build_pack.py", "--workspace", str(learner["root"]), "--cycle", "1",
            "--window", "6", "--consumed-ledger", str(consumed_ledger),
            "--streams-root", str(bank["lanes"]), "--min-targets", "1",
            "--python", sys.executable]
    with patch.object(sys, "argv", argv):
        packer.main()
    packed = [json.loads(line) for line in
              (learner["root"] / "packed-receipts.jsonl").read_text().splitlines()]
    check("F4 streams-root packing pools every lane at the learner ledger",
          len(packed) == 6
          and {p["receipt_sha256"] for p in packed} == {r["receipt_sha256"] for r in union}
          and not (collector1["root"] / "packed-receipts.jsonl").exists()
          and not (collector2["root"] / "packed-receipts.jsonl").exists()
          and not (collector3["root"] / "packed-receipts.jsonl").exists())
    sources = {r["receipt_sha256"]: r["source_lane"] for r in union}
    metadata = [json.loads(line) for line in
                (learner["packs"] / "cycle-0001" / "authority" /
                 "records.jsonl").read_text().splitlines()]
    check("F4 packed and authority metadata name source lanes on cross-stream lines",
          len(metadata) == 6
          and all(row["source_lane"] == sources[row["receipt_sha256"]] for row in metadata)
          and {p["receipt_sha256"]: p["source_lane"] for p in packed} == sources)

    authority = learner["packs"] / "cycle-0001" / "authority"
    packs = learner["packs"] / "cycle-0001" / "packs"
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=_sha(authority / "manifest.json"),
        pack_manifest_sha256=_sha(packs / "manifest.json"), sampler_key=970001,
        data_world_size=1, context_size=2048)
    data = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0,
                                  sampler_mode="epoch-permutation")
    samples = []
    for _ in range(len(data.packs)):
        sample = sampled_pack_records(data, data.next_absolute_rank_sample_index)
        data.get_boundary_aware_batch(1, device="cpu")
        samples.append(sample)
    row = {"authority_manifest_sha256": identity.authority_manifest_sha256,
           "pack_manifest_sha256": identity.pack_manifest_sha256,
           "checkpoint_sha256": "c" * 64, "sampled_packs": samples}
    exposure = lane._adopted_exposure_rows(learner, 1, row)
    lane._record_adopted_exposure(learner, 1, row)
    consumed = [json.loads(line) for line in consumed_ledger.read_text().splitlines()]
    check("F4 adopted exposure consumption carries source lanes exactly once",
          len(exposure) == len(consumed) == 6
          and len({entry["receipt_sha256"] for entry in consumed}) == 6
          and all(entry["source_lane"] == sources[entry["receipt_sha256"]]
                  for entry in consumed)
          and packer.load_consumed_ledger(consumed_ledger) == set(sources)
          and lane._block_inventory(learner, bank["lanes"]) == [])
    data.close()

    # ---- absent flag: ledger lines byte-identical, no source_lane anywhere
    single = workspace_paths(scratch / "single-stream")
    single["root"].mkdir(parents=True)
    a2 = mint(None, kind="teacher-corrected", created=2000.0, task_id="A2")
    append_receipt(single, a2)
    b2 = mint(a2["receipt_sha256"], kind="on-policy-success", created=2000.5, task_id="B2")
    append_receipt(single, b2)
    argv = ["rl_build_pack.py", "--workspace", str(single["root"]), "--cycle", "2",
            "--window", "2", "--consumed-ledger",
            str(single["root"] / "consumed-receipts.jsonl"),
            "--min-targets", "1", "--python", sys.executable]
    with patch.object(sys, "argv", argv):
        packer.main()
    lines = (single["root"] / "packed-receipts.jsonl").read_text().splitlines()
    expected_lines = [
        json.dumps({"receipt_sha256": a2["receipt_sha256"], "packed_cycle": 2, "window": 2}),
        json.dumps({"receipt_sha256": b2["receipt_sha256"], "packed_cycle": 2, "window": 2}),
    ]
    metadata2 = [json.loads(line) for line in
                 (single["packs"] / "cycle-0002" / "authority" /
                  "records.jsonl").read_text().splitlines()]
    authority2 = single["packs"] / "cycle-0002" / "authority"
    packs2 = single["packs"] / "cycle-0002" / "packs"
    identity2 = SFTSamplerIdentity(
        authority_manifest_sha256=_sha(authority2 / "manifest.json"),
        pack_manifest_sha256=_sha(packs2 / "manifest.json"), sampler_key=970001,
        data_world_size=1, context_size=2048)
    data2 = MaskedSFTPackedDataset(authority2, packs2, identity=identity2, rank=0,
                                   sampler_mode="epoch-permutation")
    samples2 = []
    for _ in range(len(data2.packs)):
        sample = sampled_pack_records(data2, data2.next_absolute_rank_sample_index)
        data2.get_boundary_aware_batch(1, device="cpu")
        samples2.append(sample)
    row2 = {"authority_manifest_sha256": identity2.authority_manifest_sha256,
            "pack_manifest_sha256": identity2.pack_manifest_sha256,
            "checkpoint_sha256": "e" * 64, "sampled_packs": samples2}
    rows2 = lane._adopted_exposure_rows(single, 2, row2)
    data2.close()
    check("F4 absent flag keeps ledger lines byte-identical without source lanes",
          lines == expected_lines
          and all("source_lane" not in record for record in metadata2)
          and len(rows2) == 2
          and all(set(entry) == {"receipt_sha256", "exposed_cycle", "sampled_pack_ids",
                                 "pack_manifest_sha256", "adopted_checkpoint_sha256"}
                  for entry in rows2)
          and not (single["root"] / "consumed-receipts.jsonl").exists())

    # ---- collector lane: receipts chain, no channel ever trains or adopts
    seed = scratch / "collector-seed.pt"
    _synthetic_checkpoint(seed, 1.0, "collector-seed")
    collector_state = init_lane_state(bank, 1, checkpoint=seed,
                                      checkpoint_sha256=_sha(seed), note="collector unit")
    collector_args = SimpleNamespace(
        lane=1, args_json=Path("/dev/null"), probe_timeout=10, pack_timeout=60,
        train_timeout=60, collector_only=True, block_mode="on",
        block_min_targets=65536, block_max_targets=131072, block_max_wait_cycles=2,
        anchor_period=1, anchor_authority_root=Path("fixture"),
        lane_train_step="sft-receipts")
    summary = {"policy_checkpoint_sha256": _sha(seed), "attempts": 1, "outcomes": []}
    guarded = []
    with patch.object(lane, "_anchor_train_row",
                      side_effect=lambda *a: guarded.append("anchor")), \
         patch.object(lane, "_sft_train_row",
                      side_effect=lambda *a: guarded.append("sft")), \
         patch.object(lane, "_run_receipts_block",
                      side_effect=lambda *a: guarded.append("block")), \
         patch.object(lane, "_pg_train_row", side_effect=lambda *a: guarded.append("pg")), \
         patch.object(lane, "_probe_adopt_train_row",
                      side_effect=lambda *a: guarded.append("probe")), \
         redirect_stdout(StringIO()) as output:
        lane._train_cycle_channels(bank, collector1, collector_state, collector_args, 1, {},
                                   collector1["root"] / "log", summary, "0", 2)
    metrics = json.loads((collector1["cycles"] / "cycle-0001" /
                          "metrics.json").read_text())
    check("F4 collector lane collects without training adopting or advancing",
          guarded == []
          and collector_state["updates_total"] == 0
          and collector_state["lineage_sha256"] == _sha(seed)
          and collector_state["status"] == "collected"
          and collector_state["superseded_lineages"] == []
          and "LANE_COLLECTOR_ONLY" in output.getvalue())
    check("F4 collector metrics record the cycle with no train row",
          metrics["train"] is None
          and metrics["trains"] == [{"train": None, "adopted": None,
                                     "probe_frame_valid": None,
                                     "train_step_channel": None}]
          and metrics["receipts"] == 2
          and metrics["receipt_kinds"] == {"teacher-corrected": 2})

    # ---- lane_run gates: collectors skip merge adoption and block accounting
    from rl_bank import read_lane_state

    run_bank = bank_paths(scratch / "run-bank")
    ensure_bank_layout(run_bank)
    ensure_pinned_validator(run_bank)
    seed2 = scratch / "run-seed.pt"
    _synthetic_checkpoint(seed2, 2.0, "run-seed")
    init_lane_state(run_bank, 0, checkpoint=seed2, checkpoint_sha256=_sha(seed2),
                    note="learner")
    init_lane_state(run_bank, 1, checkpoint=seed2, checkpoint_sha256=_sha(seed2),
                    note="collector")
    base_args = dict(bank=run_bank["root"], args_json=Path("/dev/null"),
                     teacher="fixture", teacher_model="m", max_tasks=1,
                     claim_ttl=60.0, teacher_min_interval=0.0, poll_seconds=0.05,
                     lease_wait=1, lease_retry_seconds=0.05, collect_timeout=10,
                     pack_timeout=10, train_timeout=10, probe_timeout=10,
                     max_failures=3, max_seconds=0.4, lane_train_step="sft-receipts",
                     block_min_targets=65536, block_max_targets=131072,
                     block_max_wait_cycles=48)
    learner_run = SimpleNamespace(lane=0, collector_only=False, block_mode="on",
                                  block_streams_root=None, **base_args)
    adopt_learner = MagicMock(side_effect=lambda *a, **k: None)
    with patch.object(lane, "adopt_merge_current", adopt_learner), \
         patch.object(lane, "reclaim_superseded_lineages", return_value=[]), \
         redirect_stdout(StringIO()):
        lane.lane_run(learner_run)
    learner_state = read_lane_state(lane_paths(run_bank, 0)["root"])
    check("F4 learner lane_run keeps merge polling and block accounting",
          adopt_learner.call_count >= 1
          and "block_inventory_targets" in learner_state
          and learner_state["status"] == "stopped")
    collector_run = SimpleNamespace(lane=1, collector_only=True, block_mode="on",
                                    block_streams_root=None, **base_args)
    adopt_collector = MagicMock(side_effect=lambda *a, **k: None)
    with patch.object(lane, "adopt_merge_current", adopt_collector), \
         patch.object(lane, "reclaim_superseded_lineages", return_value=[]), \
         redirect_stdout(StringIO()):
        lane.lane_run(collector_run)
    collector_run_state = read_lane_state(lane_paths(run_bank, 1)["root"])
    check("F4 collector lane_run never adopts merges nor initializes block accounting",
          adopt_collector.call_count == 0
          and all(key not in collector_run_state for key in
                  ("block_mode", "block_inventory_targets", "block_wait_cycles",
                   "block_attempts_total", "block_adopted_total"))
          and collector_run_state["lineage_sha256"] == _sha(seed2)
          and collector_run_state["updates_total"] == 0
          and collector_run_state["status"] == "stopped")


def test_pg_grid(scratch: Path) -> None:
    """F5: grid evidence, unchanged legacy bytes, and real PG replay arithmetic."""
    from contextlib import ExitStack, redirect_stdout
    from io import StringIO
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    import rl_pg_batch as builder
    import rl_policy_gradient_step as step
    from ndm.e97_onpolicy_records import canonical_json

    grid = scratch / "pg-grid" / "lanes"
    roots = [grid / f"lane-{i:02d}" for i in range(4)]
    for i, root in enumerate(roots):
        for cycle in (1, 2):
            _synthetic_episode(root, cycle, "bank-shared-id", "family",
                               passed=i % 2 == 0,
                               turn_text="Action: finish\nArguments: {}\nFinal: " +
                                         ("ok" if i % 2 == 0 else "failed sealed task"))
            summary = root / "episodes" / f"cycle-{cycle:04d}" / "collect-summary.json"
            body = json.loads(summary.read_text())
            body["policy_checkpoint_sha256"] = str(i) * 64
            summary.write_text(json.dumps(body))
            os.utime(summary, ns=(1000 + cycle * 100 + i, 1000 + cycle * 100 + i))
    _synthetic_episode(roots[1], 2, "dev", "family", passed=True, split="development")
    summary = roots[1] / "episodes" / "cycle-0002" / "collect-summary.json"
    os.utime(summary, ns=(1201, 1201))
    (grid / "lane-decoy").mkdir()
    (grid / "lane-99").mkdir()  # no episodes yet
    body = builder.collect_grid_batch(roots[0], grid, last_cycles=1, group_by="family")
    manifest = builder.bind_batch_sha(body)
    builder.verify_batch(manifest)
    check("F5 pg-grid all lanes, colliding task ids remain distinct episodes",
          len(body["entries"]) == 4 and {e["source_lane"] for e in body["entries"]} ==
          {r.name for r in roots} and body["groups"]["family"]["episodes"] == 4)
    check("F5 pg-grid family grouping, train split, pass/fail contrast",
          body["contrastive_groups"] == ["family"] and body["groups"]["family"]["passes"] == 2
          and all(e["split"] == "train" for e in body["entries"]))
    check("F5 pg-grid per-lane windows and policy provenance",
          all(e["cycle"] == 2 for e in body["entries"])
          and len({e["policy_checkpoint_sha256"] for e in body["entries"]}) == 4
          and body["lane_cycles"] == {r.name: [2] for r in roots})
    capped = builder.collect_grid_batch(roots[0], grid, last_cycles=2,
                                        group_by="family", max_episodes=2)
    check("F5 pg-grid global recency cap uses immutable summary clocks",
          [e["source_lane"] for e in capped["entries"]] == ["lane-02", "lane-03"]
          and all(e["cycle"] == 2 for e in capped["entries"]) and capped["contrast"])
    again = builder.collect_grid_batch(roots[0], grid, last_cycles=2,
                                       group_by="family", max_episodes=2)
    check("F5 pg-grid same-state mtime selection deterministic",
          capped["entries"] == again["entries"] and capped["groups"] == again["groups"])
    for root in roots:
        summary = root / "episodes" / "cycle-0002" / "collect-summary.json"
        os.utime(summary, ns=(2000, 2000))
    tied = builder.collect_grid_batch(roots[0], grid, last_cycles=1, group_by="family")
    check("F5 pg-grid recency ties numeric lane/cycle/task",
          [e["source_lane"] for e in tied["entries"]] == [r.name for r in roots])
    no_contrast = builder.collect_grid_batch(roots[0], grid, last_cycles=1,
                                             group_by="family", max_episodes=1)
    check("F5 pg-grid recency cannot manufacture contrast", not no_contrast["contrast"])
    for label, mutate in [
            ("source", lambda b: b["entries"][0].update(source_lane="../escape")),
            ("split", lambda b: b["entries"][0].update(split="development")),
            ("digest", lambda b: b["entries"][0].update(episode_sha256="f" * 64))]:
        damaged = json.loads(json.dumps(body))
        mutate(damaged)
        try:
            builder.verify_batch(builder.bind_batch_sha(damaged))
        except ValueError:
            check(f"F5 pg-grid {label} guard fail-closed", True)
        else:
            check(f"F5 pg-grid {label} guard fail-closed", False)
    try:
        builder.collect_grid_batch(roots[0], grid / "missing", last_cycles=1)
    except SystemExit:
        check("F5 pg-grid missing root fails closed", True)
    else:
        check("F5 pg-grid missing root fails closed", False)
    # Golden digest captured from the pre-F5 deployed builder. Normalize only
    # the scratch root and creation clock; every other canonical byte is bound.
    with patch.object(builder.time, "time", return_value=123):
        single = builder.collect_batch(roots[0], [1, 2], group_by="family", max_episodes=2)
    single["episodes_root"] = "legacy-root"
    check("F5 single-lane batch canonical bytes unchanged",
          builder.bind_batch_sha(single)["batch_sha256"] ==
          "db7140f5b9cc2d8732b201e03d7e457c571a7d413dd0a403d2f2c5018d54631f"
          and "streams_root" not in single and all("source_lane" not in e for e in single["entries"]))
    with patch.dict(os.environ, {"COLLECTOR_LANES": "0"}):
        check("F5 PG streams-root explicit configuration works without SFT block mode",
              lane._pg_streams_root(SimpleNamespace(block_mode="off", block_streams_root=grid)) == grid
              and lane._pg_streams_root(SimpleNamespace(block_mode="off")) is None)
    with patch.dict(os.environ, {"COLLECTOR_LANES": "3"}):
        check("F5 collector grid automatically feeds PG with block mode off",
              lane._pg_streams_root(SimpleNamespace(bank=grid.parent, block_mode="off")) == grid)

    # Drive the lane's actual build command and SFT-fallback seam.
    pg_paths = {"root": roots[0], "training": scratch / "pg-lane-training"}
    pg_args = SimpleNamespace(bank=grid.parent, lane=0, block_mode="off",
                              block_streams_root=grid, pg_window_cycles=1,
                              pg_group_by="family", pg_max_episodes=4,
                              pg_min_group=2, pg_batch_timeout=60,
                              pg_train_timeout=60, args_json=Path("/dev/null"),
                              pg_lr=2e-6, pg_kl_beta=0.01, pg_kl_guard=0.056)
    built_commands = []

    def failed_step(cmd, **kwargs):
        log = Path(cmd[cmd.index("--log-jsonl") + 1])
        log.write_text(json.dumps({"event": "guard", "guard": "realized-kl-blowout"}) + "\n")
        return SimpleNamespace(returncode=1)

    # Keep the real builder subprocess while stubbing only the PG step.
    real_run = subprocess.run

    def build_command(cmd, **kwargs):
        built_commands.append(cmd)
        real_run(cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT) + ":" +
                           os.environ.get("PYTHONPATH", "")}, check=True,
                 stdout=subprocess.DEVNULL)
    with patch.object(lane, "_spawn", side_effect=build_command), \
         patch.object(subprocess, "check_output", return_value="0" * 40), \
         patch.object(subprocess, "run", side_effect=failed_step):
        row, event = lane._pg_train_row(pg_paths, {"lineage_path": "current.pt",
                                                 "lineage_sha256": "a" * 64},
                                        pg_args, 1, os.environ, scratch / "pg-lane.log")
    lane_batch = json.loads((pg_paths["training"] / "cycle-0001" / "pg-batch.json").read_text())
    check("F5 PG lane CLI builds verified grid attempts and retains SFT guard fallback",
          "--streams-root" in built_commands[0] and len(lane_batch["entries"]) == 4
          and row is None and event["outcome"] == "guard-trip"
          and event["guard"]["guard"] == "realized-kl-blowout")

    # Exercise main() with CPU tensors, a one-parameter model, real candidate
    # loss/advantages/KL, and SGD. Only native loading/capture/CUDA are stubbed.
    import ndm.e97 as e97
    import ndm.e97_outcome_rl_candidate as candidate
    import scripts.train_e97_4b_pi_sft as trainer

    parent = scratch / "pg-parent.pt"
    parent.write_bytes(b"current-lineage")
    parent_sha = _sha(parent)
    real_device = torch.device
    real_advantages = candidate.group_advantages
    real_loss = candidate.policy_loss

    def run_step(grid_mode, *, guard=0.25, bad_ce=False, fault=None):
        events, captured, ratios, emissions = [], [], [], []
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.initialize_state_ = lambda: None
        optimizer.assert_state_offloaded = lambda: None
        optimizer.train = lambda: None
        optimizer.eval = lambda: None
        batch = json.loads(json.dumps(manifest))
        if not grid_mode:
            batch.pop("streams_root")
        # Poison generation-time traces: they must NEVER be read.
        for entry in batch["entries"]:
            entry["stored_logprobs"] = [-10000.0]
        batch = builder.bind_batch_sha({k: v for k, v in batch.items() if k != "batch_sha256"})
        batch_path = scratch / "pg-test-batch.json"
        batch_path.write_text(json.dumps(batch))
        out = scratch / f"pg-test-step-{grid_mode}-{guard}-{bad_ce}-{fault}"
        out.mkdir()

        def capture(current, prefix, generated, device, *, alignment, grad):
            events.append("capture" if not grad else "update")
            captured.append((current is model, float(current.weight.detach()), grad))
            lp = -(current.weight.square().sum()).expand(len(generated))
            return lp, float(-lp.detach().sum()) + (1 if bad_ce else 0)

        def advantages(*args):
            events.append("advantages")
            result = real_advantages(*args)
            return torch.full_like(result, float("inf")) if fault == "advantages" else result

        def loss(new, old, *args, **kwargs):
            ratios.append(torch.exp(new.detach() - old).clone())
            result = real_loss(new, old, *args, **kwargs)
            return result * float("nan") if fault == "loss" else result

        argv = ["pg", "--parent-checkpoint", str(parent), "--parent-sha256", parent_sha,
                "--args-json", "/dev/null", "--batch", str(batch_path),
                "--batch-sha256", batch["batch_sha256"], "--output-root", str(out),
                "--log-jsonl", str(out / "log"), "--pg-stream", str(out / "stream"),
                "--source-commit", "0" * 40, "--skip-checkpoint-save", "--kl-guard", str(guard)]
        error = None
        with ExitStack() as stack:
            for target, name, replacement in [
                    (sys, "argv", argv), (torch.cuda, "set_device", lambda *a: None),
                    (torch, "device", lambda *a, **k: real_device("cpu")),
                    (torch.cuda, "max_memory_allocated", lambda: 0),
                    (torch.cuda, "max_memory_reserved", lambda: 0),
                    (e97, "load_e97_checkpoint", lambda *a, **k: SimpleNamespace(model=model)),
                    (trainer, "EXPECTED_PARAMETERS", 1),
                    (trainer, "configure_precision", lambda *a: {}),
                    (trainer, "build_optimizer", lambda *a, **k: optimizer),
                    (step, "_configure_train_geometry", lambda *a: None),
                    (step, "_capture_row", capture),
                    (step, "emit", lambda path, event, **k: emissions.append((event, k))),
                    (candidate, "group_advantages", advantages), (candidate, "policy_loss", loss)]:
                stack.enter_context(patch.object(target, name, replacement))
            stack.enter_context(redirect_stdout(StringIO()))
            try:
                step.main()
            except (RuntimeError, SystemExit, ValueError) as exc:
                error = str(exc)
        return events, captured, ratios, emissions, error, out

    grid_run = run_step(True)
    single_run = run_step(False)
    check("F5 pg-grid current checkpoint pre-pass precedes advantages for every record",
          grid_run[4] is None and grid_run[0][:5] == ["capture"] * 4 + ["advantages"]
          and all(same and weight == 1 and not grad for same, weight, grad in grid_run[1][:4]))
    check("F5 pg-grid poison stale traces ignored and ratio exactly one",
          len(grid_run[2]) == 4 and all(torch.equal(r, torch.ones_like(r)) for r in grid_run[2]))
    check("F5 single-lane retains detached-forward path without pre-pass",
          single_run[4] is None and single_run[0][0] == "advantages"
          and single_run[0][1:5] == ["update"] * 4)
    grid_step = next(k for event, k in grid_run[3] if event == "step")
    single_step = next(k for event, k in single_run[3] if event == "step")
    check("F5 pg-grid matches legacy loss and realized pre/post-update KL",
          all(grid_step[k] == single_step[k] for k in
              ("loss", "grad_norm", "surrogate_part", "kl_part", "realized_kl_vs_old_mean")))
    grid_ce, single_ce = run_step(True, bad_ce=True), run_step(False, bad_ce=True)
    check("F5 pg-grid and legacy CE guard trip identically without checkpoint",
          grid_ce[4] == single_ce[4] and "consistency failed" in grid_ce[4]
          and not any(e == "checkpoint" for e, _ in grid_ce[3] + single_ce[3]))
    grid_kl, single_kl = run_step(True, guard=1e-12), run_step(False, guard=1e-12)
    check("F5 pg-grid and legacy realized-KL guard trip identically without receipt",
          grid_kl[4] == single_kl[4] and "KL blowout guard" in grid_kl[4]
          and not (grid_kl[5] / "stream").exists() and not (single_kl[5] / "stream").exists())
    for fault in ("advantages", "loss"):
        grid_fault, single_fault = run_step(True, fault=fault), run_step(False, fault=fault)
        check(f"F5 pg-grid and legacy nonfinite {fault} guard trip identically",
              grid_fault[4] == single_fault[4] and "nonfinite" in grid_fault[4]
              and not any(e == "checkpoint" for e, _ in grid_fault[3] + single_fault[3]))
    with patch.object(step, "_capture_row", side_effect=RuntimeError("captured logprob count mismatch")):
        try:
            step._replay_current_logprobs(None, [(0, {}, {"prefix": [1], "generated": [2]})],
                                         "cpu", SimpleNamespace(alignment=128))
        except RuntimeError as exc:
            check("F5 pg-grid qualified coverage guard remains fail-closed",
                  str(exc) == "captured logprob count mismatch")


def test_correction_pool(scratch: Path) -> None:
    from unittest.mock import patch
    from rl_correction_config import read_correction_config
    import rl_correction_pool as module
    root = scratch / "f7-config"
    root.mkdir()
    config = read_correction_config(root)
    check("F7 absent durable config preserves live baseline and default-OFF",
          config["teacher_model"] == "glm-5.3-flash-background" and not config["async_enabled"])
    (root / "correction-config.json").write_text(json.dumps({
        "teacher_model": "deepseek-4.1-flash-background", "async_enabled": True, "pool_width": 8}))
    check("F7 durable config survives independent reread", read_correction_config(root)["pool_width"] == 8)
    for bad in ({"pool_width": 0}, {"pool_width": True}, {"async_enabled": "yes"}, {"unknown": 1}):
        (root / "correction-config.json").write_text(json.dumps(bad))
        try:
            read_correction_config(root)
        except ValueError:
            check("F7 invalid durable correction config fails closed " + str(bad), True)
        else:
            check("F7 invalid config accepted", False)
    real_popen = subprocess.Popen
    # Actual OS processes exercise capacity, partial/crashed results and teardown;
    # substitute only child workload, never the production pool implementation.
    script = ('import json,time,sys; from pathlib import Path; '
              's=json.loads(Path(sys.argv[1]).read_text()); time.sleep(s["delay"]); '
              'Path(sys.argv[2]).write_text(json.dumps({"passed":True,"value":s["value"]})); '
              'sys.exit(s.get("exit",0))')
    def spawn(command, **kwargs):
        return real_popen([sys.executable, "-c", script,
                          command[command.index("--spec")+1], command[command.index("--result")+1]], **kwargs)
    with patch.object(module.subprocess, "Popen", side_effect=spawn):
        pool = module.CorrectionPool(2, scratch / "f7-pool", timeout_s=2)
        first, _ = pool.submit({"delay": .4, "value": "first"})
        second, _ = pool.submit({"delay": .01, "value": "second"})
        seq, error = pool.submit({"delay": 0, "value": "exhausted"})
        check("F7 pool exhaustion is explicit and never silently queued",
              seq is None and error["error"] == "pool exhausted" and len(pool.jobs) == 2)
        immutable = pool.directory / "spec-0000.json"
        check("F7 admitted correction specs immutable", immutable.stat().st_mode & 0o222 == 0)
        time.sleep(.15)
        ready = pool.poll()
        check("F7 completion ordering reports ready correction without waiting for earlier one",
              len(ready) == 1 and ready[0][0] == second and ready[0][1]["value"] == "second")
        time.sleep(.45)
        check("F7 earlier slow correction completes exactly once", pool.poll()[0][0] == first and pool.poll() == [])
        crash_dir = scratch / "f7-crash-metrics"
        crash_dir.mkdir()
        call = {"model": "unit", "latency_s": .1, "attempt": 0}
        (crash_dir / "correction-api.jsonl").write_text(json.dumps(call)+"\n{partial")
        crashed, _ = pool.submit({"delay": .01, "value": "partial", "exit": 7,
                                  "task_dir": str(crash_dir)})
        time.sleep(.15)
        ready = pool.poll()
        check("F7 crash mid-correction cannot admit even a present result file",
              ready[0][0] == crashed and not ready[0][1]["passed"] and "exit 7" in ready[0][1]["error"]
              and ready[0][1]["metrics"]["calls"] == [call] and ready[0][1]["metrics_partial_tail"])
        late, _ = pool.submit({"delay": 10, "value": "late"})
        proc = pool.jobs[late][0]
        cancelled = pool.close()
        check("F7 late completion after cycle close is fenced and child is reaped",
              cancelled == [late] and proc.poll() is not None and pool.poll() == [])
        check("F7 closed cycle refuses further admission", pool.submit({})[1]["error"] == "pool closed")
        timed = module.CorrectionPool(1, scratch / "f7-timeout", timeout_s=.05)
        timed.submit({"delay": 10, "value": "never"})
        time.sleep(.1)
        ready = timed.poll()
        check("F7 correction deadline bounded and publishes no success", ready[0][1]["error"] == "correction timeout")
        timed.close()
        twelve = module.CorrectionPool(12, scratch / "f7-twelve-drain", timeout_s=2)
        for index in range(12):
            twelve.submit({"delay": .05+.01*index, "value": index})
        drain_started = time.monotonic()
        finalized = []
        while twelve.jobs:
            finalized.extend(twelve.poll())
            if twelve.jobs:
                time.sleep(.01)
        drain_s = time.monotonic()-drain_started
        twelve.close()
        (scratch / "F7-pool-drain-test.json").write_text(json.dumps({
            "workers": 12, "terminal_drain_s": drain_s,
            "qualification": "actual OS processes with CPU fixture sleeps, NOT LunarRoute latency"})+"\n")
        check("F7 twelve-worker terminal fence measures bounded real drain gap",
              len(finalized) == 12 and .05 <= drain_s < 2, f"drain_s={drain_s:.6f}")

    # Real worker, real Linux parent-death signal; no GPU/model/API workload.
    import signal
    owner_script = ('import sys,time; from pathlib import Path; '
                    'from rl_correction_pool import CorrectionPool; '
                    'p=CorrectionPool(1,Path(sys.argv[1])); seq,_=p.submit({}); '
                    'Path(sys.argv[2]).write_text(str(p.jobs[seq][0].pid)); time.sleep(60)')
    child_id = scratch / "f7-owner-child.pid"
    owner = real_popen([sys.executable, "-c", owner_script, str(scratch / "f7-owner-death"),
                       str(child_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic()+10
        ready = False
        while time.monotonic() < deadline:
            if child_id.exists():
                child = int(child_id.read_text())
                status_path = Path(f"/proc/{child}/status")
                if status_path.exists():
                    caught = next(line.split()[1] for line in status_path.read_text().splitlines()
                                  if line.startswith("SigCgt:"))
                    if int(caught, 16) & (1 << (signal.SIGTERM-1)):
                        ready = True
                        break
            time.sleep(.01)
        check("F7 real worker installs owner-death fence before correction imports", ready)
        owner.kill()
        owner.wait(timeout=5)
        stopped = False
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            status_path = Path(f"/proc/{child}/status")
            if not status_path.exists() or "State:\tZ" in status_path.read_text():
                stopped = True
                break
            time.sleep(.01)
        check("F7 collect SIGKILL terminates isolated correction owner", stopped)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_async_correction_collect(scratch: Path) -> None:
    """Real lane admission/single-writer finalization, isolated CPU fakes only."""
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    import rl_bank_lane as lane
    root = scratch / "f7-async-collect"
    bank = bank_paths(root)
    ensure_bank_layout(bank)
    seed = root / "seed.pt"
    _synthetic_checkpoint(seed, 1.0, "seed")
    init_lane_state(bank, 0, checkpoint=seed, checkpoint_sha256=_sha(seed), note="unit")
    (root / "correction-config.json").write_text(json.dumps({"async_enabled": True, "pool_width": 2}))
    tasks = [{"task_id": f"f7-{i}", "task_sha256": str(i)*64,
              "body": {"template": "fixture", "prompt": "x", "split": "train",
                       "receipt_eligible": True, "workspace_files": {}}} for i in range(3)]
    events, emitted, retired, specs = [], [], [], []
    record = {"status": "stopped", "generations": [], "source_messages": [], "reason": "fixture"}
    def episode(**kwargs):
        events.append("attempt")
        (kwargs["episode_dir"] / "episode-private.json").write_text(json.dumps(record))
        return record
    class Pool:
        def __init__(self, width, directory):
            self.jobs = {}
            self.closed = False
        def submit(self, spec):
            events.append("admit")
            specs.append(spec)
            if len(self.jobs) == 2:
                return None, {"passed": False, "error": "pool exhausted"}
            sequence = len(self.jobs)
            self.jobs[sequence] = spec
            return sequence, None
        def poll(self):
            if events.count("attempt") < 3 or len(specs) < 3:
                return []
            rows = []
            for seq in reversed(list(self.jobs)):
                rows.append((seq, {"passed": True, "grade": {"passed": True},
                    "fresh_solve": True, "teacher_record": {"status": "finished",
                    "close_verified": True, "prefix": None, "reason": None, "generations": []}}))
            self.jobs.clear()
            return rows
        def close(self):
            self.closed = True
    def receipt(**kwargs):
        emitted.append(kwargs)
        events.append("receipt")
        return {"receipt_sha256": str(len(emitted))*64, "episode": {"targets": 10}}
    def retire(bank, task, **kwargs):
        retired.append((task["task_id"], kwargs["outcome"]))
    args = SimpleNamespace(bank=root, lane=0, cycle=1, args_json=Path("/dev/null"),
                           teacher="fixture", teacher_model=None, preclaimed=None,
                           max_tasks=3, claim_ttl=60, teacher_min_interval=0)
    with ExitStack() as stack:
        replacements = {"load_pilot": SimpleNamespace(Metrics=lambda: SimpleNamespace(calls=[])),
            "load_curriculum": SimpleNamespace(SYSTEM="fixture"),
            "tool_manifest": {"pi_bin": "/dev/null", "model_visible_tools": []},
            "load_policy_engine": None, "make_policy_generate": None,
            "heartbeat_claim": True, "stream_tail_digest": None}
        for attribute, value in replacements.items():
            stack.enter_context(patch.object(lane, attribute, return_value=value))
        stack.enter_context(patch.object(lane, "CorrectionPool", Pool))
        stack.enter_context(patch.object(lane, "claim_pool_task", side_effect=tasks))
        stack.enter_context(patch.object(lane, "run_episode", side_effect=episode))
        stack.enter_context(patch.object(lane, "grade_episode", return_value=(False, {})))
        stack.enter_context(patch.object(lane, "_receipt_from_episode", side_effect=receipt))
        stack.enter_context(patch.object(lane, "append_receipt", side_effect=lambda paths, row: row["receipt_sha256"]))
        stack.enter_context(patch.object(lane, "retire_task", side_effect=retire))
        lane.bank_collect(args)
    check("F7 next attempt runs before any asynchronous correction receipt",
          events[:6] == ["attempt", "admit", "attempt", "admit", "attempt", "admit"])
    check("F7 single writer chains out-of-order receipts using publication-time tail",
          [x["task"]["task_id"] for x in emitted] == ["f7-1", "f7-0"]
          and emitted[0]["prev"] is None and emitted[1]["prev"] == "1"*64)
    summary = json.loads((lane_paths(bank, 0)["episodes"] / "cycle-0001/collect-summary.json").read_text())
    check("F7 cycle summary reports terminal correction drain separately",
          summary["correction_terminal_drain_s"] >= 0)
    check("F7 exhausted task remains honestly uncorrected with no receipt",
          summary["receipts"] == 2 and summary["correction_pool_exhaustions"] == 1
          and summary["outcomes"][2]["receipt"] is None
          and summary["outcomes"][2]["correction"]["reason"] == "pool exhausted"
          and next(row for task_id, row in retired if task_id == "f7-2")["stage"] == "failed")
    check("F7 correction specs preserve exact task/attempt and separate workspaces",
          len({x["workspace"] for x in specs}) == 3 and all(x["policy_record"] == record for x in specs))


def test_correction_spec_binding(scratch: Path) -> None:
    """Hash-bound archived attempts must match immutable correction snapshots."""
    from ndm.e97_onpolicy_records import canonical_json
    from rl_bank_lane import execute_correction_spec
    binding_dir = scratch / "f7-snapshot-binding" / "attempt"
    binding_dir.mkdir(parents=True)
    record = {"generations": [], "source_messages": []}
    (binding_dir / "episode-private.json").write_text(canonical_json(record)+"\n")
    spec = {"task_dir": str(binding_dir.parent), "episode_artifacts": {
        "attempt_episode_sha256": _sha(binding_dir / "episode-private.json")},
        "policy_record": {**record, "reason": "altered snapshot"}}
    try:
        execute_correction_spec(spec)
        rejected_snapshot = False
    except ValueError as exc:
        rejected_snapshot = "snapshot differs" in str(exc)
    check("F7 execute_correction_spec rejects a snapshot differing from hash-bound attempt", rejected_snapshot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "smoke" / "bank-unit")
    sub = parser.add_subparsers(dest="command")
    worker = sub.add_parser("claim-worker")
    worker.add_argument("bank")
    worker.add_argument("lane", type=int)
    worker.add_argument("out")
    args = parser.parse_args()
    if args.command == "claim-worker":
        _claim_worker(args.bank, args.lane, args.out)
        return
    scratch = args.scratch
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    test_pool(scratch)
    test_merge(scratch)
    test_trigger()
    test_lake(scratch / "bank")
    test_adoption(scratch / "adopt-bank")
    test_reclaim(scratch / "reclaim-bank")
    test_pg_channel(scratch)
    test_receipt_exposure(scratch)
    test_lane_channel_chain(scratch)
    test_invalid_probe_rejection(scratch)
    test_fresh_solve_first_action(scratch)
    test_fresh_solve_collect(scratch)
    test_no_signal_exposure_skip(scratch)
    test_rejected_anchor_chain(scratch)
    test_channel_checkpoint_paths(scratch)
    test_block_training(scratch)
    test_parallel_collection(scratch)
    test_pg_grid(scratch)
    test_correction_pool(scratch)
    test_correction_spec_binding(scratch)
    test_async_correction_collect(scratch)
    print(json.dumps({"schema": "emender-rl-loop-bank-unit-test-v1",
                     "passed": len(PASSED), "tests": PASSED}, sort_keys=True))


if __name__ == "__main__":
    main()
