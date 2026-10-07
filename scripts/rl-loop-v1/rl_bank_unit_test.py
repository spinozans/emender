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
            return {"status": "finished", "generations": [],
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
         patch('rl_receipts.walk_stream', return_value=inventory):
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
    print(json.dumps({"schema": "emender-rl-loop-bank-unit-test-v1",
                     "passed": len(PASSED), "tests": PASSED}, sort_keys=True))


if __name__ == "__main__":
    main()
